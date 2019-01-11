import gensim
import networkx as nx
import theano
from theano import config
from collections import OrderedDict
import timeit
import six.moves.cPickle as pickle
import downhill

import data_utils
import tprnn_model
from datetime import datetime

# sys.path.append('../../')
# LOG_DIR = "log/"
# log_file = LOG_DIR + '%s_%s_%s_%s.log' % (dataset_name, str(today.year), str(today.month), str(today.day))
# log_level = logging.INFO
# logging.basicConfig(filename=log_file, level=log_level, format='%(asctime)s - %(levelname)s: %(message)s',
#                     datefmt='%m/%d/%Y %H:%M:%S')



today = datetime.today()
dataset_name = 'christianity'
is_next_node_prediction = False



from eval_metrics import *

def numpy_floatX(data):
    return np.asarray(data, dtype=config.floatX)


def ortho_weight(ndim):
    W = np.random.randn(ndim, ndim)
    u, s, v = np.linalg.svd(W)
    return u.astype(config.floatX)


def init_params(options):
    """
    Initializes values of shared variables.
    """
    params = OrderedDict()

    # word embedding, shape = (n_words, dim_proj)
    randn = np.random.randn(options['n_words'], options['dim_proj'])
    params['Wemb'] = (0.1 * randn).astype(config.floatX)

    # shape = dim_proj * (4*dim_proj)
    lstm_W = np.concatenate([ortho_weight(options['dim_proj']),
                             ortho_weight(options['dim_proj']),
                             ortho_weight(options['dim_proj']),
                             ortho_weight(options['dim_proj'])], axis=1)
    params['lstm_W'] = lstm_W

    # shape = dim_proj * (4*dim_proj)
    lstm_U = np.concatenate([ortho_weight(options['dim_proj']),
                             ortho_weight(options['dim_proj']),
                             ortho_weight(options['dim_proj']),
                             ortho_weight(options['dim_proj'])], axis=1)
    params['lstm_U'] = lstm_U

    lstm_b = np.zeros((4 * options['dim_proj'],))
    params['lstm_b'] = lstm_b.astype(config.floatX)

    # decoding matrix for external influences
    randn = np.random.randn(options['dim_proj'], options['n_words'])
    params['W_ext'] = (0.1 * randn).astype(config.floatX)
    dec_b = np.zeros(options['n_words'])
    params['b_ext'] = dec_b.astype(config.floatX)

    return params


def init_tparams(params):
    '''
    Set up Theano shared variables.
    '''
    tparams = OrderedDict()
    for kk, pp in params.items():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams


def unzip(zipped):
    """
    When we pickle the model. Needed for the GPU stuff.
    """
    new_params = OrderedDict()
    for kk, vv in zipped.items():
        new_params[kk] = vv.get_value()
    return new_params


def load_params(path, params):
    pp = np.load(path)
    for kk, vv in params.items():
        if kk not in pp:
            raise Warning('%s is not in the archive' % kk)
        params[kk] = pp[kk]

    return params


def evaluate(f_prob, test_loader, k_list=[10, 50, 100]):
    '''
    Evaluates trained model.
    '''
    n_batches = len(test_loader)
    y = None
    y_prob = None
    target_labels = []
    for _ in range(n_batches):
        batch_data = test_loader()
        y_ = batch_data[-2]
        target = batch_data[-1]
        y_prob_ = f_prob(*batch_data[:-2])
        # excludes activated nodes when predicting.
        for i, p in enumerate(y_prob_):
            length = int(np.sum(batch_data[1][:, i]))
            sequence = batch_data[0][: length, i]
            assert y_[i] not in sequence, str(sequence) + str(y_[i])
            p[sequence] = 0.
            y_prob_[i, :] = p / float(np.sum(p))
        target_labels.extend(target)
        if y_prob is None:
            y_prob = y_prob_
            y = y_
        else:
            y = np.concatenate((y, y_), axis=0)
            y_prob = np.concatenate((y_prob, y_prob_), axis=0)
    relevance_scores = []
    M_list = []
    for i in range(0, y_prob.shape[0]):
        target = y[i]
        target_set = target_labels[i]
        predictedUsers = list(np.argsort(y_prob[i])[-200:]) # sorted order.
        predictedUsers.reverse()
        if is_next_node_prediction:
            relevance_score = np.equal(predictedUsers, target)
        else:
            relevance_score = np.isin(predictedUsers, target_set)
        #print (relevance_score)
        relevance_scores.append(relevance_score)
        M_list.append(len(target_set))
    #return metrics.portfolio(y_prob, y, k_list=k_list)
    for k in k_list:
        print ("MAP@",k, MAP(relevance_scores, k, M_list))
        print ("NDCG@", k, mean_NDCG_at_k(relevance_scores, k))
        print ("Precision@", k, mean_precision_at_k(relevance_scores, k))
        print ("Recall@", k, mean_recall_at_k(relevance_scores, k, M_list))

        logging.info("MAP@ %d : %f" %(k, MAP(relevance_scores,k, M_list)))
        logging.info("NDCG@ %d : %f" %(k, mean_NDCG_at_k(relevance_scores,k)))
        logging.info("Precision@ %d : %f" %(k, mean_precision_at_k(relevance_scores,k)))
        logging.info("Recall@ %d : %f" %(k, mean_recall_at_k(relevance_scores,k, M_list)))

    print ("MRR", MRR(relevance_scores))
    logging.info("MRR: %f" % MRR(relevance_scores))
    return [-1]


def save_embedding(embed, index_node, path, binary=False):
    learned_embed = gensim.models.keyedvectors.Word2VecKeyedVectors(embed.shape[1])
    learned_embed.add(index_node, embed)
    learned_embed.save_word2vec_format(fname=path, binary=binary, total_vec=len(index_node))


def train(data_dir='../author_graph_dataset/',
          dim_proj=100, # was 512
          maxlen=100,  # was 50
          batch_size=1024, #256,
          keep_ratio=1.,
          shuffle_data=True,
          learning_rate=0.001,
          global_steps= 50000, # 20000, # was 50000
          disp_freq=200,
          save_freq=20000,
          #test_freq=50,
          saveto_file='params.npz',
          embed_file='topolstm_embedding.txt',
          weight_decay=0.0005,
          reload_model=False,
          train=True):


    # Topo-LSTM model training.
    options = locals().copy() # copy params
    saveto = data_dir + saveto_file

    # loads graph
    G, node_index, index_node = data_utils.load_graph(data_dir)
    print(nx.info(G))
    options['n_words'] = len(node_index)

    print(options)

    # creates and initializes shared variables.
    print('Initializing variables...')
    params = init_params(options)

    if reload_model:
        print('reusing saved model.')
        load_params(saveto, params)

    tparams = init_tparams(params)

    # builds Topo-LSTM model
    print('Building model...')
    model = tprnn_model.build_model(tparams, options)

    # print('Loading test data from eval_set.txt ...')
    # test_examples = data_utils.load_eval_examples(data_dir,
    #                                               dataset='eval_set',
    #                                               node_index=node_index,
    #                                               maxlen=maxlen,
    #                                               G=G)
    # test_loader = data_utils.Loader(test_examples, options=options, train=False)
    # print('Loaded %d test examples' % len(test_examples))

    if train:
        # prepares training data.
        print('Loading train data...')
        train_examples = data_utils.load_examples(data_dir,
                                                  dataset='active_sequence',
                                                  keep_ratio=options['keep_ratio'],
                                                  node_index=node_index,
                                                  maxlen=maxlen,
                                                  G=G)

        train_loader = data_utils.Loader(train_examples, options=options)
        print('Loaded %d training examples.' % len(train_examples))

        # compiles updates.
        optimizer = downhill.build(algo='adam',
                                   loss=model['cost'],
                                   params=list(tparams.values()),
                                   inputs=model['data'])

        updates = optimizer.get_updates(max_gradient_elem=5.,
                                        learning_rate=learning_rate)

        f_update = theano.function(model['data'],
                                   model['cost'],
                                   updates=list(updates))

        # training loop.
        start_time = timeit.default_timer()

        # downhill.minimize(
        #     loss=cost,
        #     algo='adam',
        #     train=train_loader,
        #     # inputs=input_list + [labels],
        #     # params=tparams.values(),
        #     # patience=0,
        #     max_gradient_clip=1,
        #     # max_gradient_norm=1,
        #     learning_rate=learning_rate,
        #     monitors=[('cost', cost)],
        #     monitor_gradients=False)

        n_examples = len(train_examples)
        batches_per_epoch = n_examples // options['batch_size'] + 1
        n_epochs = global_steps // batches_per_epoch + 1
        global_step = 0
        cost_history = []

        print("# num of examples", n_examples)
        print("# epochs", n_epochs)
        print("# batches per epoch", batches_per_epoch)
        for ep in range(n_epochs):
            print("Ep", ep + 1)
            for b in range(batches_per_epoch):
                print("Ep{}/{}, batch{}/{}\n".format(ep+1, n_epochs, b+1, batches_per_epoch))
                # actually run and update
                cost = f_update(*train_loader())
                cost_history += [cost]

                if global_step % disp_freq == 0:
                    print('global step %d, cost: %f' % (global_step, cost))
                    logging.info("global step %d, cost: %f" % (global_step, cost))

                # dump model parameters.
                if global_step % save_freq == 0:
                    params = unzip(tparams)
                    np.savez(saveto, **params)
                    pickle.dump(options, open('%s.pkl' % saveto, 'wb'), -1)

                # evaluate on test data.
                # if global_step % test_freq == 0:
                #     scores = evaluate(model['f_prob'], test_loader)
                #     print('eval scores: ', scores)
                #     end_time = timeit.default_timer()
                #     logging.info('time used: %d seconds.' % (end_time - start_time))

                global_step += 1

    #scores = evaluate(model['f_prob'], test_loader)
    #print('eval scores: ', scores)

    print('Save embedding...')
    embed = model['embs'].get_value()
    save_embedding(embed, index_node, path=embed_file)




if __name__ == '__main__':
    train()
    print("end of main")
