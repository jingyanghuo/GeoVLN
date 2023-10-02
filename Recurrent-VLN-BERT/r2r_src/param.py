import argparse
import os
import torch

class Param:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="")

        # General
        self.parser.add_argument('--test_only', type=int, default=0, help='fast mode for testing')
        self.parser.add_argument('--iters', type=int, default=100000, help='training iterations')
        self.parser.add_argument('--name', type=str, default='default', help='experiment id')
        self.parser.add_argument('--vlnbert', type=str, default='prevalent', help='oscar or prevalent')
        self.parser.add_argument('--train', type=str, default='auglistener')
        self.parser.add_argument('--description', type=str, default='no description\n')
        self.parser.add_argument('--log_every', type=int, default=2000, help='image height')
        self.parser.add_argument('--batchSize', type=int, default=8)

        self.parser.add_argument("--apex", action="store_const", default=False, const=True)
        self.parser.add_argument("--visualize", action="store_const", default=False, const=True)

        self.parser.add_argument("--mp_end", action="store_const", default=False, const=True)
        self.parser.add_argument("--xdyt", action="store_const", default=False, const=True)
        self.parser.add_argument("--slot_attn", action="store_const", default=False, const=True)
        self.parser.add_argument("--slot_dropout", type=float, default=0, help='dropout rate for slot attention')
        self.parser.add_argument("--slot_ignore_end", action="store_const", default=False, const=True)
        self.parser.add_argument("--slot_share_qk", action="store_const", default=False, const=True)
        self.parser.add_argument("--slot_noise", action="store_const", default=False, const=True)
        self.parser.add_argument("--slot_residual", action="store_const", default=False, const=True)
        self.parser.add_argument("--slot_local_mask", action="store_const", default=False, const=True)
        self.parser.add_argument('--slot_local_mask_h', type=int, default=3, help='local mask horizontal span')
        self.parser.add_argument('--slot_local_mask_v', type=int, default=3, help='local mask vertical span')
        self.parser.add_argument("--discriminator", action="store_const", default=False, const=True)

        # sub-instruction
        self.parser.add_argument("--sub_instr", action="store_const", default=False, const=True)
        self.parser.add_argument('--max_subs', type=int, default=16, help='max number of sub instructions')

        self.parser.add_argument("--trar_mask", action="store_const", default=False, const=True)
        self.parser.add_argument('--trar_pooling', type=str, default='attention')


        # clip
        self.parser.add_argument("--clip_after_encoder", action="store_const", default=False, const=True)
        self.parser.add_argument('--clip_weight', type=float, default=None, help="the learning rate")

        # Augmented Paths from
        self.parser.add_argument("--aug", default=None)

        # simulator
        self.parser.add_argument('--image_w', type=int, default=640, help='image width')
        self.parser.add_argument('--image_h', type=int, default=480, help='image height')
        self.parser.add_argument("--render_image", action="store_const", default=False, const=True)

        # maxpooling feature
        self.parser.add_argument('--max_pool_feature', type=str, default=None, help='path of the max pooled feature')

        # object match
        self.parser.add_argument("--object", action="store_const", default=False, const=True)
        self.parser.add_argument('--top_N_obj', type=int, default=8)
        self.parser.add_argument("--nerf_pe", action="store_const", default=False, const=True)
        self.parser.add_argument('--match_type', type=str, default='max', help='instruction and object tag match type, [max, mean]')

        # learning rate
        self.parser.add_argument('--warm_up_epochs', type=int, default=5, help='warmup')
        self.parser.add_argument('--lr_adjust_type', type=str, default='cosine', help='learning rate adjust type')
        self.parser.add_argument("--warm_steps", type=int, default=10)
        self.parser.add_argument("--decay_start", type=int, default=20)
        self.parser.add_argument("--decay_intervals", type=int, default=15)
        self.parser.add_argument("--lr_decay", type=float, default=0.2)
        self.parser.add_argument('--optim', type=str, default='adamW')  # rms, adam
        self.parser.add_argument('--lr', type=float, default=0.00001, help="the learning rate")
        self.parser.add_argument('--decay', dest='weight_decay', type=float, default=0.01)
        self.parser.add_argument('--gaussian_lr', type=float, default=0.00001, help="the learning rate")
        self.parser.add_argument('--pg_lr', type=float, default=0.0001, help="the learning rate")
        self.parser.add_argument("--reset_lr", action="store_const", default=False, const=True)

        # Data preparation
        self.parser.add_argument('--maxInput', type=int, default=80, help="max input instruction")
        self.parser.add_argument('--maxAction', type=int, default=15, help='Max Action sequence')
        self.parser.add_argument('--ignoreid', type=int, default=-100)
        self.parser.add_argument('--feature_size', type=int, default=2048)

        # Load the model from
        self.parser.add_argument("--load", default=None, help='path of the trained model')
        self.parser.add_argument("--loadOptim", action="store_const", default=False, const=True)

        # Listener Model Config
        self.parser.add_argument("--zeroInit", dest='zero_init', action='store_const', default=False, const=True)
        self.parser.add_argument("--mlWeight", dest='ml_weight', type=float, default=0.20)
        self.parser.add_argument("--pgWeight", dest='pg_weight', type=float, default=None)
        self.parser.add_argument("--teacherWeight", dest='teacher_weight', type=float, default=1.)
        self.parser.add_argument("--features", type=str, default='places365')

        # Dropout Param
        self.parser.add_argument('--dropout', type=float, default=0.5)
        self.parser.add_argument('--featdropout', type=float, default=0.3)

        # Submision configuration
        self.parser.add_argument("--submit", type=int, default=0)

        # Training Configurations
        self.parser.add_argument('--feedback', type=str, default='sample',
                            help='How to choose next position, one of ``teacher``, ``sample`` and ``argmax``')
        self.parser.add_argument('--teacher', type=str, default='final',
                            help="How to get supervision. one of ``next`` and ``final`` ")

        # Model hyper params:
        self.parser.add_argument("--angleFeatSize", dest="angle_feat_size", type=int, default=128)
        self.parser.add_argument('--epsilon', type=float, default=0.1)
        # A2C
        self.parser.add_argument("--gamma", default=0.9, type=float)
        self.parser.add_argument("--normalize", dest="normalize_loss", default="total", type=str, help='batch or total')

        self.args = self.parser.parse_known_args()[0]

        if self.args.optim == 'rms':
            print("Optimizer: Using RMSProp")
            self.args.optimizer = torch.optim.RMSprop
        elif self.args.optim == 'adam':
            print("Optimizer: Using Adam")
            self.args.optimizer = torch.optim.Adam
        elif self.args.optim == 'adamW':
            print("Optimizer: Using AdamW")
            self.args.optimizer = torch.optim.AdamW
        elif self.args.optim == 'sgd':
            print("Optimizer: sgd")
            self.args.optimizer = torch.optim.SGD
        else:
            assert False

param = Param()
args = param.args

args.description = args.name
args.IMAGENET_FEATURES = 'img_features/ResNet-152-imagenet.tsv'
args.log_dir = 'snap/%s' % args.name

if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)
DEBUG_FILE = open(os.path.join('snap', args.name, "debug.log"), 'w')
