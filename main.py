from model.train import load_data, model_train
from utility.utility import history_plot

import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True, type=str, help="drmm, pacrr, pacrr_drmm")
ap.add_argument("--bert", required=False, type=bool, help="bert")
ap.add_argument("--lr", required=False, type=float, help="learning rate")
ap.add_argument("--batch", required=False, type=float, help="batch size")
ap.add_argument("--epoch", required=False, type=int, help="total epoch count")
args = ap.parse_args()

if args.bert:
    firstk = 13
    lq = 8
else:
    firstk = 8
    lq = 6
lg = 5
nf = 32
ns = 2

if args.model == 'drmm':
    from model.drmm import Gen_DRMM_Model
    model = Gen_DRMM_Model(bert=args.bert)
elif args.model == 'pacrr':
    from model.pacrr import Gen_PACRR_Model
    model = Gen_PACRR_Model(firstk, lq, lg, nf, ns, bert=args.bert)
elif args.model == 'pacrr_drmm':
    from model.pacrr_drmm import Gen_PACRR_DRMM_Model
    model = Gen_PACRR_DRMM_Model(firstk, lq, lg, nf, ns, bert=args.bert)
    
batch_size = 128
if args.batch:
    batch_size = args.batch

total_epoch_count = 100
if args.epoch:
    total_epoch_count = args.epoch

lr = 0.1
if args.lr:
    lr = args.lr


if __name__ == "__main__":
    data = load_data()
    train, model_history, model_metric = model_train(data, lr, model, batch_size, total_epoch_count, bert=args.bert, model_name=args.model)
    history_plot(model_history, model_metric, batch_size, df=train, save=True)