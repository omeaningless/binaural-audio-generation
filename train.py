import pdb
import os
import os.path as osp
import time
import torch
import torchvision
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.networksf21 import VisualNet, VisualNetDilated, AssoConv, APNet1, weights_init, AudioNet1,netD3,Stereo,Attention
from tensorboardX import SummaryWriter

def compute_loss(output, f,flag, audio_mix, loss_criterion, prefix="stereo", weight=44):
    loss_dict = {}
    
    key = '{}_loss'.format(prefix)
    loss_dict[key] = loss_criterion(output['binaural_spectrogram'], output['audio_gt'].detach()) * weight

    if 'pred_left' in output:
        fusion_loss1 = loss_criterion(2*output['pred_left']-audio_mix[:,:,:-1,:], output['audio_gt'].detach())
        fusion_loss2 = loss_criterion(audio_mix[:,:,:-1,:]-2*output['pred_right'], output['audio_gt'].detach())
        key = '{}_loss_fusion'.format(prefix)
        loss_dict[key] = (fusion_loss1 / 2  + fusion_loss2 / 2) *weight


    key = '{}_loss_class'.format(prefix)
    flag = flag.view(flag.size(0),-1)
    loss_dict[key]=torch.nn.BCELoss()(f,flag)
    loss_dict

    return loss_dict

def save_model(net_audio, net_stereo, net_visual, net_fusion, net_class, net_att1, net_att2,opt, suffix=''):
    torch.save(net_visual.module.state_dict(), osp.join('.', opt.checkpoints_dir, opt.name, 'visual_{}.pth'.format(suffix)))
    torch.save(net_audio.module.state_dict(), osp.join('.', opt.checkpoints_dir, opt.name, 'audio_{}.pth'.format(suffix)))
    torch.save(net_stereo.module.state_dict(),
               osp.join('.', opt.checkpoints_dir, opt.name, 'stereo_{}.pth'.format(suffix)))
    torch.save(net_class.module.state_dict(), osp.join('.', opt.checkpoints_dir, opt.name, 'class_{}.pth'.format(suffix)))
    torch.save(net_att1.module.state_dict(),
               osp.join('.', opt.checkpoints_dir, opt.name, 'att1_{}.pth'.format(suffix)))
    torch.save(net_att2.module.state_dict(),
               osp.join('.', opt.checkpoints_dir, opt.name, 'att2_{}.pth'.format(suffix)))
    if net_fusion is not None:
        torch.save(net_fusion.module.state_dict(), osp.join('.', opt.checkpoints_dir, opt.name, 'fusion_{}.pth'.format(suffix)))

def create_optimizer(nets, opt):
    (net_visual, net_audio, net_stereo, net_fusion,net_class, net_att1, net_att2) = nets
    param_groups = [
                {'params': net_visual.parameters(), 'lr': opt.lr_visual},
                {'params': net_audio.parameters(), 'lr': opt.lr_audio},
                {'params': net_stereo.parameters(), 'lr': opt.lr_stereo},
                {'params': net_class.parameters(), 'lr': opt.lr_class},
                {'params': net_att1.parameters(), 'lr': opt.lr_att1},
                {'params': net_att2.parameters(), 'lr': opt.lr_att2}
            ]
    if net_fusion is not None:
        param_groups.append({'params': net_fusion.parameters(), 'lr': opt.lr_fusion})
    if opt.optimizer == 'sgd':
        return torch.optim.SGD(param_groups, momentum=opt.beta1, weight_decay=opt.weight_decay)
    elif opt.optimizer == 'adam':
        return torch.optim.Adam(param_groups, betas=(opt.beta1,0.999), weight_decay=opt.weight_decay)

def decrease_learning_rate(optimizer, decay_factor=0.94):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay_factor

#used to display validation loss
def display_val(nets, loss_criterion, writer, index, data_loader_val, opt, return_key):
    (net_visual, net_audio, net_stereo, net_fusion, net_class, net_att1, net_att2) = nets
    val_loss_log = {} 
    with torch.no_grad():
        for i, val_data in enumerate(data_loader_val):
            if i < opt.validation_batches:
                val_total_loss = {}
                audio_diff = val_data['audio_diff_spec'].to(opt.device)
                audio_mix = val_data['audio_mix_spec'].to(opt.device)
                visual_input = val_data['frame'].to(opt.device)
                left = val_data['left'].to(opt.device)
                right = val_data['right'].to(opt.device)
                flag = val_data['flag'].to(opt.device)
                vfeat = net_visual(visual_input)
                vfeat1 = net_att1(vfeat)
                vfeat2 = net_att2(vfeat)
                upfeatures, output = net_audio(audio_diff, audio_mix, vfeat1,  return_upfeatures=True)
                output.update(net_fusion(audio_mix, vfeat1, upfeatures))
                mid = net_stereo(left, right)
                f = net_class(mid,vfeat2)

                val_total_loss.update(compute_loss(output, f, flag, audio_mix, loss_criterion, prefix='stereo', weight=opt.stereo_loss_weight))


                for loss_name, loss_value in val_total_loss.items():
                    if loss_name not in val_loss_log:
                        val_loss_log[loss_name] = [loss_value.item()]
                    else:
                        val_loss_log[loss_name].append(loss_value.item())
            else:
                break

    avg_val_loss_log = {}
    print("--- Val loss info ---")
    for key, value in val_loss_log.items():
        avg_value = sum(value) / len(value)
        avg_val_loss_log[key] = avg_value 
        print("val_{}: {:.3f}".format(key, avg_value))
        if opt.tensorboard:
            writer.add_scalar('data/val_{}'.format(key), avg_value, index)
    print("\n")

    return avg_val_loss_log[return_key] 

#parse arguments
opt = TrainOptions().parse()
opt.device = torch.device("cuda")

#construct data loader
data_loader = CreateDataLoader(opt)

#create validation set data loader if validation_on option is set
if opt.validation_on:
    #temperally set to val to load val data
    opt.mode = 'val'
    data_loader_val = CreateDataLoader(opt)
    opt.mode = 'train' #set it back

if opt.tensorboard:
    from tensorboardX import SummaryWriter
    writer = SummaryWriter(comment=opt.name)
else:
    writer = None

## build network
# visual net
original_resnet = torchvision.models.resnet18(pretrained=True)
if opt.visual_model == 'VisualNet':
    net_visual = VisualNet(original_resnet)
elif opt.visual_model == 'VisualNetDilated':
    net_visual = VisualNetDilated(original_resnet)
else:
    raise TypeError("please input correct visual model type")

if len(opt.weights_visual) > 0:
    print('Loading weights for visual stream')
    net_visual.load_state_dict(torch.load(opt.weights_visual), strict=True)



# audio net
net_audio = AudioNet1(
    ngf=opt.unet_ngf,
    input_nc=opt.unet_input_nc,
    output_nc=opt.unet_output_nc,
    norm_mode=opt.norm_mode
)
net_audio.apply(weights_init)
if len(opt.weights_audio) > 0:
    print('Loading weights for audio stream')
    net_audio.load_state_dict(torch.load(opt.weights_audio), strict=True)



net_stereo = Stereo(
    ngf=opt.unet_ngf,
    input_nc=4,
    output_nc=opt.unet_output_nc,
    norm_mode=opt.norm_mode
)
net_stereo.apply(weights_init)
if len(opt.weights_stereo) > 0:
    print('Loading weights for audio2 stream')
    net_stereo.load_state_dict(torch.load(opt.weights_audio2), strict=True)


net_class = netD3()
net_class.apply(weights_init)
if len(opt.weights_class) > 0:
    print('Loading weights for class stream')
    net_class.load_state_dict(torch.load(opt.weights_class), strict=True)

net_att1 = Attention()
net_att1.apply(weights_init)
if len(opt.weights_att1) > 0:
    print('Loading weights for att1 stream')
    net_att1.load_state_dict(torch.load(opt.weights_att1), strict=True)

net_att2 = Attention()
net_att2.apply(weights_init)
if len(opt.weights_att2) > 0:
    print('Loading weights for att2 stream')
    net_att2.load_state_dict(torch.load(opt.weights_att2), strict=True)


# fusion net

net_fusion = APNet1(norm_mode=opt.norm_mode)
if net_fusion is not None and len(opt.weights_fusion) > 0:
    net_fusion.load_state_dict(torch.load(opt.weights_fusion), strict=True)

# data parallel
nets = (net_visual, net_audio, net_stereo, net_fusion, net_class, net_att1, net_att2)
net_visual.to(opt.device)
net_visual = torch.nn.DataParallel(net_visual, device_ids=opt.gpu_ids)
net_audio.to(opt.device)
net_audio = torch.nn.DataParallel(net_audio, device_ids=opt.gpu_ids)
net_stereo.to(opt.device)
net_stereo = torch.nn.DataParallel(net_stereo, device_ids=opt.gpu_ids)
net_class.to(opt.device)
net_class =torch.nn.DataParallel(net_class,device_ids=opt.gpu_ids)
net_att1.to(opt.device)
net_att1 =torch.nn.DataParallel(net_att1, device_ids=opt.gpu_ids)
net_att2.to(opt.device)
net_att2 = torch.nn.DataParallel(net_att2,
                                 device_ids=opt.gpu_ids)
net_fusion.to(opt.device)
net_fusion = torch.nn.DataParallel(net_fusion, device_ids=opt.gpu_ids)

# set up optimizer
optimizer = create_optimizer(nets, opt)

# set up loss function
if opt.loss_mode == 'mse':
    loss_criterion = torch.nn.MSELoss()
elif opt.loss_mode == 'l1':
    loss_criterion = torch.nn.L1Loss()
else:
    raise TypeError("Please use correct loss mode")
if len(opt.gpu_ids) > 0:
    loss_criterion.cuda(opt.gpu_ids[0])

# initialization
total_steps = 0
data_loading_time = []
model_forward_time = []
model_backward_time = []
loss_log = {}
best_err = float("inf")

for epoch in range(1, opt.niter+1):
    torch.cuda.synchronize()
    epoch_start_time = time.time()

    if opt.measure_time:
        iter_start_time = time.time()
    for i, data in enumerate(data_loader):
        if opt.measure_time:
            torch.cuda.synchronize()
            iter_data_loaded_time = time.time()

        total_steps += opt.batchSize

        total_loss = {}
        # forward

        audio_diff = data['audio_diff_spec'].to(opt.device)
        audio_mix = data['audio_mix_spec'].to(opt.device)
        visual_input = data['frame'].to(opt.device)
        left = data['left'].to(opt.device)
        right = data['right'].to(opt.device)
        flag = data['flag'].to(opt.device)
        vfeat = net_visual(visual_input)
        vfeat1 = net_att1(vfeat)
        vfeat2 = net_att2(vfeat)

        upfeatures, output = net_audio(audio_diff, audio_mix, vfeat1,return_upfeatures=True)
        output.update(net_fusion(audio_mix, vfeat1, upfeatures))
        mid = net_stereo(left, right)
        f = net_class(mid, vfeat2)

        total_loss.update(compute_loss(output, f, flag, audio_mix, loss_criterion, prefix='stereo', weight=opt.stereo_loss_weight))

        # parse loss
        loss = sum(_value for _key, _value in total_loss.items() if 'loss' in _key)
        for loss_name, loss_value in total_loss.items():
            if loss_name not in loss_log:
                loss_log[loss_name] = [loss_value.item()]
            else:
                loss_log[loss_name].append(loss_value.item())

        if opt.measure_time:
            torch.cuda.synchronize()
            iter_data_forwarded_time = time.time()

        # update optimizer
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if opt.measure_time:
            iter_model_backwarded_time = time.time()
            data_loading_time.append(iter_data_loaded_time - iter_start_time)
            model_forward_time.append(iter_data_forwarded_time - iter_data_loaded_time)
            model_backward_time.append(iter_model_backwarded_time - iter_data_forwarded_time)

        if total_steps // opt.batchSize % opt.display_freq == 0:
            print('Display training progress at (epoch %d, total_steps %d)' % (epoch, total_steps))
            for key, value in loss_log.items():
                avg_value = sum(value) / len(value)
                print("{}: {:.3f}".format(key, avg_value))
                if opt.tensorboard:
                    writer.add_scalar('data/{}'.format(key), avg_value, total_steps) 
            print("\n")
            loss_log = {} 
            if opt.measure_time:
                print('average data loading time: ' + str(sum(data_loading_time)/len(data_loading_time)))
                print('average forward time: ' + str(sum(model_forward_time)/len(model_forward_time)))
                print('average backward time: ' + str(sum(model_backward_time)/len(model_backward_time)))
                data_loading_time = []
                model_forward_time = []
                model_backward_time = []

        if total_steps // opt.batchSize % opt.save_latest_freq == 0:
            print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
            save_model(net_audio, net_stereo, net_visual, net_fusion, net_class, net_att1, net_att2, opt, suffix='latest')

        if total_steps // opt.batchSize % opt.validation_freq == 0 and opt.validation_on:
            net_visual.eval()
            net_audio.eval()
            net_stereo.eval()
            net_class.eval()
            net_att1.eval()
            net_att2.eval()
            if net_fusion is not None:
                net_fusion.eval()
            opt.mode = 'val'
            print('Display validation results at (epoch %d, total_steps %d)' % (epoch, total_steps))
            nets = (net_visual, net_audio, net_stereo, net_fusion, net_class, net_att1, net_att2)
            val_err = display_val(nets, loss_criterion, writer, total_steps, data_loader_val, opt, return_key=opt.val_return_key)
            net_visual.train()
            net_audio.train()
            net_stereo.train()
            net_class.train()
            net_att1.train()
            net_att2.train()

            if net_fusion is not None:
                net_fusion.train()
            opt.mode = 'train'
            #save the model that achieves the smallest validation error
            if val_err < best_err:
                best_err = val_err
                print('saving the best model (epoch %d, total_steps %d) with validation error %.3f\n' % (epoch, total_steps, val_err))
                save_model(net_audio, net_stereo, net_visual, net_fusion, net_class,net_att1, net_att2,opt, suffix='best')

        if opt.measure_time:
            iter_start_time = time.time()

    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, total_steps %d' % (epoch, total_steps))
        save_model(net_audio, net_stereo, net_visual, net_fusion, net_class, net_att1, net_att2, opt, suffix=str(epoch))

    #decrease learning rate 6% every opt.learning_rate_decrease_itr epochs
    if opt.learning_rate_decrease_itr > 0 and epoch % opt.learning_rate_decrease_itr == 0:
        decrease_learning_rate(optimizer, opt.decay_factor)
        print('decreased learning rate by ', opt.decay_factor)
