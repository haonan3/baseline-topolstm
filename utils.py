from __future__ import print_function
import numpy as np
import networkx as nx
import scipy.sparse as sp
import random
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape
    def to_tuple_list(matrices):
        # Input is a list of matrices.
        coords = []
        values = []
        shape = [len(matrices)]
        for i in range(0, len(matrices)):
            mx = matrices[i]
            if not sp.isspmatrix_coo(mx):
                mx = mx.tocoo()
            # Create proper indices - coords is a numpy array of pairs of indices.
            coords_mx = np.vstack((mx.row, mx.col)).transpose()
            z = np.array([np.ones(coords_mx.shape[0])*i]).T
            z = np.concatenate((z,coords_mx), axis = 1)
            z = z.astype(int)
            #coords.extend(z.tolist())
            coords.extend(z)
            values.extend(mx.data)

        shape.extend(matrices[0].shape)
        shape = np.array(shape).astype("int64")
        values = np.array(values).astype("float32")
        coords = np.array(coords)
        # print ("insider", len(coords), len(values), shape)
        return coords, values, shape

    if isinstance(sparse_mx, list) and isinstance(sparse_mx[0], list):
        # Given a list of lists, convert it into a list of tuples.
        for i in range(0, len(sparse_mx)):
            sparse_mx[i] = to_tuple_list(sparse_mx[i])

    elif isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx

def normalize_graph_gcn(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_to_tuple(adj_normalized)

def to_one_hot(labels, N, multilabel=False):
    """In: list of (nodeId, label) tuples, #nodes N
       Out: N * |label| matrix"""
    ids, labels = zip(*labels)
    lb = MultiLabelBinarizer()
    if not multilabel:
        labels = [[x] for x in labels]
    lbs = lb.fit_transform(labels)
    encoded = np.zeros((N, lbs.shape[1]))
    for i in range(len(ids)):
        encoded[ids[i]] = lbs[i]
    return encoded

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

extra_tokens = ['_GO', 'EOS']

def get_data_set(dataset_str, cascades, timestamps, maxlen=None, min_seeds_size=1, mode = 'test'):
    dataset = []
    dataset_times = []
    eval_set = []
    eval_set_times = []
    for cascade in cascades:
        if maxlen is None or len(cascade) < maxlen:
            dataset.append(cascade)
        else:
            dataset.append(cascade[0:maxlen])  # truncate

    for ts_list in timestamps:
        if maxlen is None or len(ts_list) < maxlen:
            dataset_times.append(ts_list)
        else:
            dataset_times.append(ts_list[0:maxlen]) # truncate

    for cascade, ts_list in zip(dataset, dataset_times):
        assert len(cascade) == len(ts_list)
        for j in range(1, len(cascade)):
            seedSet = cascade[0:j]
            seedSet_times = ts_list[0:j]
            # nextUser = cascade[j]
            remain = cascade[j:]
            remain_times = ts_list[j:]
            # for k in range(j+1, len(cascade)):
            #     if len(cascade[j-1:k])>= 5:
            #         eval_set.append((cascade[j-1:k],cascade[k:]))
            # For creating evaluation set and our training set.
            if mode =='train' and len(seedSet) >= min_seeds_size:
                eval_set.append((seedSet, remain))
                eval_set_times.append((seedSet_times, remain_times))
            if (mode == 'test' or mode =='val') and len(seedSet) >= min_seeds_size and len(seedSet) <= FLAGS.max_seeds_size:
                eval_set.append((seedSet, remain))
                eval_set_times.append((seedSet_times, remain_times))
    if mode =='test':
        with open("data/{}/{}".format(dataset_str, "eval_set.txt"), 'w') as f:
            for (example,target) in eval_set:
                #random.shuffle(example)
                f.write(','.join(map(str,example))+"\t"+','.join(map(str,target))+"\n")
                #f.write(','.join(map(str,cascade))+"\n")
    if mode == 'val':
        with open("data/{}/{}".format(dataset_str, "val_set.txt"), 'w') as f:
            for (example,target) in eval_set:
                f.write(','.join(map(str,example))+"\t"+','.join(map(str,target))+"\n")
    print ("# {} examples {}".format(mode,len(eval_set)))
    return eval_set, eval_set_times

def load_graph(dataset_str):
    """Load graph."""
    print ("Loading graph", dataset_str)
    G = nx.Graph()  # TODO: undirected?
    with open("data/{}/{}".format(dataset_str, "graph_new.txt"), 'rb') as f:
        nu = 0
        for line in f:
            nu += 1
            if nu == 1:
                # assuming first line contains number of nodes, edges.
                nNodes, nEdges = [int(x) for x in line.strip().split()]
                for i in range(nNodes):
                    G.add_node(i)
                continue
            s, t = [int(x) for x in line.strip().split()]
            G.add_edge(s, t)
    A = nx.adjacency_matrix(G)
    print("# nodes", nNodes, "# edges", nEdges, A.shape)
    global start_token, end_token
    start_token = A.shape[0]+extra_tokens.index('_GO')	# start_token = 0
    end_token = A.shape[0]+extra_tokens.index('EOS')	# end_token = 1
    return A

def load_cascades(dataset_str, mode='train'):
    """Load data."""
    print ("Loading cascade", dataset_str, "mode", mode)
    cascades = []
    global avg_diff
    avg_diff =0.0
    time_stamps = []
    path = mode +str("_new.txt")
    with open("data/{}/{}".format(dataset_str, path), 'rb') as f:
        for line in f:
            if len(line) <1:
                continue
            line = list(map(float, line.split()))
            start = int(line[0])
            rest = line[1:]
            cascade = [start]
            cascade.extend(list(map(int, rest[::2])))
            time_stamp = [0]
            time_stamp.extend(rest[1::2])
            #for i in range(len(time_stamp)-1, 0, -1):
            #    time_stamp[i] -= time_stamp[i-1]

            #time_stamp[0] = 0
            #avg_diff += sum(time_stamp)/len(time_stamp)
            cascades.append(cascade)
            time_stamps.append(time_stamp)
            #print (len(cascade), len(time_stamp))
            #print (cascade, time_stamp)

    #avg_diff /= len(time_stamps)
    return cascades, time_stamps #, node_idx, idx_node


def prepare_batch_sequences(input_sequences, target_sequences, batch_size):
    # Split based on batch_size
    assert (len(input_sequences) == len(target_sequences))
    if len(input_sequences) % batch_size ==0:
        num_batch = len(input_sequences)// batch_size
    else:
        num_batch = len(input_sequences) // batch_size + 1
    batches_x  = []
    batches_y = []
    N = len(input_sequences)
    for i in range(0, num_batch):
        start = i*batch_size
        end = min((i+1)*batch_size,N)
        batches_x.append(input_sequences[start:end])
        batches_y.append(target_sequences[start:end])
    return (batches_x, batches_y)

def prepare_batch_graph(A, batch_size):
    N = A.shape[0]
    num_batch = N // batch_size + 1
    print ("batch size", batch_size)
    print ("num batches", num_batch)
    random_ordering = np.random.permutation(N)
    batches = []
    batches_indices = []
    for i in range(0, num_batch):
        start = i*batch_size
        end = min((i+1)*batch_size,N)
        batch_indices = random_ordering[start:end]
        batch = A[batch_indices,:]
        batches.append(batch.toarray())
        batches_indices.append(batch_indices)
    return batches, batches_indices


def prepare_sequences(examples, examples_times, maxlen=None, attention_batch_size=1, mode='train'):
    seqs_x = list(map(lambda seq_t: (seq_t[0][(-1)*maxlen:],seq_t[1]), examples))
    times_x = list(map(lambda seq_t: (seq_t[0][(-1)*maxlen:],seq_t[1]), examples_times))
    # add padding.
    lengths_x = [len(s[0]) for s in seqs_x]
    lengths_y = [len(s[1]) for s in seqs_x]

    if len(seqs_x) % attention_batch_size != 0 and (mode == 'test' or mode == 'val'):
        # Note: this is required to ensure that each batch is full-sized -- else the
        # data may not be split perfectly while evaluation.
        x_batch_size = (1 + len(seqs_x) // attention_batch_size)* attention_batch_size
        lengths_x.extend([1]*(x_batch_size- len(seqs_x)))
        lengths_y.extend([1]*(x_batch_size- len(seqs_x)))

    x_lengths = np.array(lengths_x).astype('int32')
    maxlen_x = maxlen
    # mask input with start token (n_nodes + 1) to work with embedding_lookup
    x = np.ones((len(lengths_x), maxlen_x)).astype('int32') * start_token
    # mask target with -1 so that tf.one_hot will return a zero vector for padded nodes
    y = np.ones((len(lengths_y), maxlen_x)).astype('int32') * -1  # we u
    x_times = np.ones((len(lengths_x), maxlen_x)).astype('int32') * -1
    y_times = np.ones((len(lengths_y), maxlen_x)).astype('int32') * -1
    mask = np.ones_like(x)
    for idx, (s_x, t) in enumerate(seqs_x):
        start = -1*lengths_x[idx]
        end_y = lengths_y[idx]
        x[idx, start:] = s_x
        y[idx, :end_y] = t
        mask[idx, :start] = 0

    for idx, (s_x, t) in enumerate(times_x):
        start = -1*lengths_x[idx]
        end_y = lengths_y[idx]
        x_times[idx, start:] = s_x
        y_times[idx, :end_y] = t

    # print (x[0], x_lengths[0], y[0], mask[0])
    return x, x_lengths, y, mask, x_times, y_times

# batch preparation of a given sequence pair for training
## TODO : Re-write this properly.
# def prepare_train_sequences(seqs_x, seqs_y, maxlen=None, trim=True, mode='train'):
#     # seqs_x, seqs_y: a list of sentences
#     lengths_x = [len(s) for s in seqs_x]
#     lengths_y = [len(s) for s in seqs_y]

#     if maxlen is not None:
#         new_seqs_x = []
#         new_seqs_y = []
#         new_lengths_x = []
#         new_lengths_y = []

#         if trim==False:
#             for l_x, s_x, l_y, s_y in zip(lengths_x, seqs_x, lengths_y, seqs_y):
#                 if l_x <= maxlen and l_y <= maxlen:
#                     new_seqs_x.append(s_x)
#                     new_lengths_x.append(l_x)
#                     new_seqs_y.append(s_y)
#                     new_lengths_y.append(l_y)
#         elif trim==True:
#             for l_x, s_x, l_y, s_y in zip(lengths_x, seqs_x, lengths_y, seqs_y):
#                 new_seqs_x.append(s_x[0:maxlen])
#                 new_lengths_x.append(min(l_x, maxlen))
#                 new_seqs_y.append(s_y[0:(maxlen-1)]) ## Note: output sequence is of size one lesser.
#                 new_lengths_y.append(min(l_y, maxlen-1))

#         lengths_x = new_lengths_x
#         seqs_x = new_seqs_x
#         lengths_y = new_lengths_y
#         seqs_y = new_seqs_y

#         if len(lengths_x) < 1 or len(lengths_y) < 1:
#             return None, None, None, None

#     x_batch_size = len(seqs_x)

#     if len(seqs_x) % FLAGS.rnn_batch_size != 0 and mode == 'test':
#         x_batch_size = (1 + len(seqs_x) // FLAGS.rnn_batch_size)* FLAGS.rnn_batch_size
#         lengths_x.extend([1]*(x_batch_size- len(seqs_x)))
#         lengths_y.extend([0]*(x_batch_size-len(seqs_y)))

#     x_lengths = np.array(lengths_x).astype('int32')
#     y_lengths = np.array(lengths_y).astype('int32')
#     maxlen_x = np.max(x_lengths)
#     maxlen_y = np.max(y_lengths)

#     # Pad it till max length of all seqs.
#     x = np.ones((x_batch_size, maxlen_x)).astype('int32') * end_token
#     y = np.ones((x_batch_size, maxlen_y)).astype('int32') * end_token

#     for idx, [s_x, s_y] in enumerate(zip(seqs_x, seqs_y)):
#         x[idx, :lengths_x[idx]] = s_x
#         y[idx, :lengths_y[idx]] = s_y
#     ## TODO: here.
#     print ("prepare here...", len(x), len(y), len(x_lengths), len(y_lengths))
#     #print (x_lengths, y_lengths)
#     return x, x_lengths, y, y_lengths
