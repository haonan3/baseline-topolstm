import os
import codecs
import networkx as nx
import numpy as np
import pickle
from theano import config


def load_graph(data_dir):
    # loads nodes observed in any cascade.
    node_file = os.path.join(data_dir, 'seen_nodes.txt')
    with open(node_file, 'rb') as f:
        seen_nodes = [x.strip() for x in f]

    # builds node index
    index_node = []
    node_index = {}
    for i, v in enumerate(seen_nodes):
        node_index[v] = i
        index_node.append(int(v))

    # loads graph
    graph_file = os.path.join(data_dir, 'author-1900-2020-link-all_copy.txt')

    G = nx.Graph()
    G.name = data_dir
    n_nodes = len(node_index)
    G.add_nodes_from(range(n_nodes))  # the node index is continuous
    with open(graph_file, 'rb') as f:
        for line in f:
            token = ",".encode('utf-8')
            u, v = line.strip().split(token)
            if (u in node_index) and (v in node_index): # check whether the edge is valid
                u = node_index[u]
                v = node_index[v]
                G.add_edge(u, v)

    return G, node_index, index_node


def convert_cascade_to_examples_train(sequence, G=None, inference=False, load_eval=False):
    length = len(sequence)

    # grows the series of dags incrementally.
    examples = []
    dag = nx.DiGraph()
    for i, node in enumerate(sequence):
        # grows the DAG.
        prefix = sequence[: i + 1]
        dag.add_node(node)
        predecessors = set(G[node]) & set(prefix)
        dag.add_edges_from([(v, node) for v in predecessors])
        # (optional) adds chronological edges
        if i > 0:
            dag.add_edge(sequence[i - 1], node)

        if i == length - 1 and not inference:
            return examples

        if i < length - 1 and inference:
            continue

        if i < length -2 and load_eval:
            continue
        # compiles example from DAG.
        node_pos = {v: i for i, v in enumerate(prefix)}
        prefix_len = len(prefix)
        topo_mask = np.zeros((prefix_len, prefix_len), dtype=np.int)
        for i_v, v in enumerate(prefix):
            i_p = [node_pos[x] for x in dag.predecessors(v)]
            topo_mask[i_v, i_p] = 1

        if not inference:
            label = sequence[i + 1]
        else:
            label = None

        example = {'sequence': prefix,
                   'topo_mask': topo_mask,
                   'label': label}

        if not inference:
            examples.append(example)
        else:
            return example



def convert_cascade_to_examples(sequence, target_set=None, G=None, inference=False, load_eval=False):

    length = len(sequence)
    # grows the series of dags incrementally.
    examples = []
    dag = nx.DiGraph()
    for i, node in enumerate(sequence):
        # 1/grows the DAG.
        prefix = sequence[: i + 1]
        dag.add_node(node)
        predecessors = set(G[node]) & set(prefix)
        dag.add_edges_from([(v, node) for v in predecessors])
        
        # 2/(optional) adds chronological edges
        if i > 0:
            dag.add_edge(sequence[i - 1], node)

        if i == length - 1 and not inference:
            return examples

        if i < length - 1 and inference:
            continue

        if i < length -2 and load_eval:
            continue
        
        # 3/compiles example from DAG.
        node_pos = {v: i for i, v in enumerate(prefix)}
        prefix_len = len(prefix)
        topo_mask = np.zeros((prefix_len, prefix_len), dtype=np.int)
        for i_v, v in enumerate(prefix):
            i_p = [node_pos[x] for x in dag.predecessors(v)]
            topo_mask[i_v, i_p] = 1

        if not inference:
            label = sequence[i + 1]
        else:
            label = None

        example = {'sequence': prefix,
                   'topo_mask': topo_mask,
                   'label': label,
                   'target_labels': target_set}

        if not inference:
            examples.append(example)
        else:
            return example


## TODO: Added code here.
def load_eval_examples(data_dir, dataset=None, G=None, node_index=None, maxlen=None, keep_ratio=1.):
    # pkl_path = os.path.join(data_dir, dataset + '.pkl')
    filename = os.path.join(data_dir, dataset + '.txt')
    examples = []
    with codecs.open(filename, 'r', encoding='utf-8') as input_file:
        for line_index, line in enumerate(input_file):
            line = line.strip()
            cascade = line.split("\t")[0].split(",")
            nextNodes = line.split("\t")[1].split(",")
            nextNode = nextNodes[0]
            cascade.append(nextNode)
            sequence = [node_index[bytes(x,'utf-8')] for x in cascade]
            targets = [node_index[bytes(x,'utf-8')] for x in nextNodes]
            #nextNode = node_index[x]
            #sequence.append(nextNode)
            # Get only one sub example --
            sub_examples = convert_cascade_to_examples(sequence, targets, G=G, load_eval=True)
            #print (sub_examples[-1]['sequence'])
            #print (sub_examples[-1]) # include only the last one.
            examples.append(sub_examples[-1])

    # pickle.dump(examples, open(pkl_path, 'wb'))

    n_samples = len(examples)
    print (n_samples)
    indices = np.random.choice(n_samples, int(n_samples * keep_ratio), replace=False)
    sampled_examples = [examples[i] for i in indices]
    return sampled_examples


def load_examples(data_dir, dataset=None, G=None, node_index=None, maxlen=None, keep_ratio=1.):
    """
    Load the train/dev/test data
    Return: list of example tuples
    """

    # loads cascades
    filename = os.path.join(data_dir, dataset + '.txt')
    print("load data from file: " + filename)
    examples = []
    with codecs.open(filename, 'r', encoding='utf-8') as input_file:
        for line_index, line in enumerate(input_file):
            # parses the input line to sequence
            sequence = line.strip().split(' ')
            # TODO : This needs to checked and fixed !!
            #if maxlen is not None and len(sequence) > maxlen:
            #    continue
            #else:
            sequence = sequence[:maxlen]
            sequence = [node_index[bytes(x,'utf-8')] for x in sequence]
            sub_examples = convert_cascade_to_examples_train(sequence, G=G)
            examples.extend(sub_examples)


    n_samples = len(examples)
    print (n_samples)
    indices = np.random.choice(n_samples, int(n_samples * keep_ratio), replace=False)
    sampled_examples = [examples[i] for i in indices]
    return sampled_examples


def prepare_minibatch(tuples, train_flag=True, inference=False, options=None):
    '''
    produces a mini-batch of data in format required by model.
    '''
    seqs = [t['sequence'] for t in tuples]
    lengths = map(len, seqs)
    n_timesteps = max(lengths)
    n_samples = len(tuples)
    lengths = list(map(len, seqs))
    # prepare sequences data
    seqs_matrix = np.zeros((n_timesteps, n_samples)).astype('int32')
    for i, seq in enumerate(seqs):
        seqs_matrix[: lengths[i], i] = seq

    # prepare topo-masks data
    topo_masks = [t['topo_mask'] for t in tuples]
    topo_masks_tensor = np.zeros((n_timesteps, n_samples, n_timesteps)).astype(config.floatX)
    for i, topo_mask in enumerate(topo_masks):
        topo_masks_tensor[: lengths[i], i, : lengths[i]] = topo_mask

    # prepare sequence masks
    seq_masks_matrix = np.zeros((n_timesteps, n_samples)).astype(config.floatX)
    for i, length in enumerate(lengths):
        seq_masks_matrix[: length, i] = 1.

    # prepare labels data
    if train_flag is True:
        labels = [t['label'] for t in tuples]
        labels_vector = np.array(labels).astype('int32')
        return (seqs_matrix,
                seq_masks_matrix,
                topo_masks_tensor,
                labels_vector)
    else:
        #labels_vector = None
        labels = [t['label'] for t in tuples]
        labels_vector = np.array(labels).astype('int32')
        target_labels = [t['target_labels'] for t in tuples]
        return (seqs_matrix,
                seq_masks_matrix,
                topo_masks_tensor,
                labels_vector, target_labels)



class Loader:
    def __init__(self, data, options=None, train=True):
        self.batch_size = options['batch_size']
        self.idx = 0
        self.data = data
        self.shuffle = options['shuffle_data']
        self.n = len(data)
        self.n_words = options['n_words']
        self.indices = np.arange(self.n, dtype="int32")
        self.options = options
        self.train = train

    def __len__(self):
        return len(self.data) // self.batch_size + 1

    def __call__(self):
        if self.shuffle and self.idx == 0:
            np.random.shuffle(self.indices)

        batch_indices = self.indices[self.idx: self.idx + self.batch_size]
        batch_examples = [self.data[i] for i in batch_indices]

        self.idx += self.batch_size
        if self.idx >= self.n:
            self.idx = 0

        return prepare_minibatch(tuples = batch_examples,
                                 inference=False, 
                                 train_flag=self.train,
                                 options=self.options)
