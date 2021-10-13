from jsonargparse import ArgumentParser, ActionConfigFile

parser = ArgumentParser()
# parser.add_argument('--lev1.opt1', default='from default 1')
# parser.add_argument('--lev1.opt2', default='from default 2')
# parser.add_argument('--cfg', action=ActionConfigFile)
# cfg = parser.parse_args(['--lev1.opt1', 'from arg 1',
#                              '--cfg', 'example.yaml',
#                              '--lev1.opt2', 'from arg 2'])

parser.add_argument('--cfg', action=ActionConfigFile)
# parser.add_argument('--dataset', type=str)
# parser.add_argument('--root', type=str)
# parser.add_argument('--n-ways', type=int)
# parser.add_argument('--k-shots', type=int)
# parser.add_argument('--q-shots', type=int)
# parser.add_argument('--inner-adapt-steps-train', type=int)
# parser.add_argument('--inner-adapt-steps-test', type=int)
parser.add_argument('--inner-lr', type=float)
# parser.add_argument('--meta-lr', type=float)
# parser.add_argument('--meta-batch-size', type=int)
# parser.add_argument('--iterations', type=int)
# parser.add_argument('--wt-ce', type=float)
# parser.add_argument('--klwt', type=str)
# parser.add_argument('--rec-wt', type=float)
# parser.add_argument('--beta-l', type=float)
# parser.add_argument('--beta-s', type=float)
# parser.add_argument('--experiment', type=str)
# parser.add_argument('--order', type=str)
parser.add_argument('--device', type=str)
parser.add_argument('--download', type=str)

args = parser.parse_args()

args.device = True
print(args.device, args.download, args.inner_lr)