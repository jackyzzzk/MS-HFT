import argparse
import configparser

def parse_args():
    parser = argparse.ArgumentParser(description='Autoformer & Transformer family for Time Series Forecasting')
    parser.add_argument('--model', type=str, default='stmodel', help='model details')
    parser.add_argument('--n', type=str, default='ccmip6', help='model details')
    # enso_data
    parser.add_argument('--data_path', type=str, default='/data1/kjzhang/data/ENSO_Dataset/cmip6/tianchi_data/', help='location of data')
    parser.add_argument('--cmip', type=str, default='cmip6', help='data name')
    parser.add_argument('--feature', type=str, default='sst_t300', help='feature of data')
    #parser.add_argument('--feature_num', type=int, default=2, help='feature number')
    parser.add_argument('--batch_size', type=int, default=2, help='batch size of train input data')
    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
    parser.add_argument('--seq_len', type=int, default=12, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=9, help='input sequence length')
    parser.add_argument('--pred_len', type=int, default=24, help='prediction sequence length')
    parser.add_argument('--f_num', type=int, default=2)
    parser.add_argument('--gap', type=int, default=5)
    parser.add_argument('--all_ratio', type=float, default=0.7, help='loss ratio')
    # hgcn
    parser.add_argument('--hyout_dim', type=int, default=16, help='hypergraph embedding dim')
    parser.add_argument('--hy_dim', type=int, default=32, help='hyper d_model dim')
    parser.add_argument('--k', type=int, default=3, help='time embedding dim1')
    parser.add_argument('--window_size', type=str, default=[3*4], help='time embedding dim2')
    parser.add_argument('--hyper_num', type=str, default=[1500,100], help='time embedding dim3')
    parser.add_argument('--hyper_attn', type=bool, default=True, help= 'if hyper use attn')
    parser.add_argument('--hyper_heads', type=int, default=4, help='hyper attn heads')
    parser.add_argument('--windows_ratio', type=float, default=0.7, help='window_ratio')
    parser.add_argument('--num_caps', type=int, default=4, help='capsule numbers')


    # spatial models
    parser.add_argument('--num_nodes', type=int, default=24*48, help='number of nodes')#1152
    parser.add_argument('--num_route', type=int, default=3, help='route number of capsule')
    parser.add_argument('--spa_drop', type=float, default=0.5, help='route iteration number')

    parser.add_argument('--freq', type=str, default='m',
                        help='freq for time features encoding, options:'
                             '[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], '
                             'you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--d_layers', type=int, default=2, help='num of decoder layers')
    # atten
    parser.add_argument('--t_dropout', type=float, default=0.05, help='fully connected dropout')
    parser.add_argument('--attn_enhance', type=int, default=1, help='attn_enhance')
    parser.add_argument('--attn_softmax_flag', type=int, default=1, help='attn_softmax_flag')
    parser.add_argument('--attn_weight_plus', type=int, default=0, help='attn_weight_plus')
    parser.add_argument('--attn_outside_softmax', type=int, default=0, help='attn_outside_softmax')
    parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')

    parser.add_argument('--d_model', type=int, default=256, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--embed_size', type=int, default=8, help='embed_size')


    # basic config
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='optimizer learning rate')
    parser.add_argument('--interval', type=int, default=10, help='interval')
    parser.add_argument('--lr_decay_rate', type=int, default=0.1, help='decay_rate')
    parser.add_argument('--lr_decay_steps', type=str, default='50,90', help='decay steps')
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--use_multi_gpu', type=bool, default=False, help='multi_gpu')
    parser.add_argument('--device_ids', type=str, default='3,4,5,6')
    parser.add_argument('--gpu', type=int, default=1, help='gpu')
    #parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')
    parser.add_argument('--best_path', type=str, default='./best_model/', help='path to save best model')

    args = parser.parse_args()
    return args