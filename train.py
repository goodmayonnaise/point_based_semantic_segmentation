from time import time

from utils.logs import AverageMeter, ProgressMeter
from utils.metrics import IOUEval

import torch 

class Training():     
    def __init__(self, model, epochs, train_loader, val_loader, optimizer, criterion, criterion_feat, nclasses,
                 scheduler, model_path, earlystop, device, metrics, writer_train, writer_val):
        super(Training, self).__init__()

        self.model = model
        self.epochs = epochs
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.criterion_feat = criterion_feat
        self.nclasses = nclasses
        self.scheduler = scheduler
        self.model_path = model_path
        self.earlystop = earlystop
        self.device = device
        self.metrics = metrics
        self.writer_train = writer_train
        self.writer_val = writer_val
        self.iou = IOUEval(n_classes=nclasses, ignore=0)

        self.b_t = AverageMeter('time', ':6.3f')
        self.loss_run = AverageMeter('Loss', ':.4f')
        self.miou_run = AverageMeter('mIoU', ':.4f')

        
    def reset(self):
        self.b_t.reset()
        self.loss_run.reset()
        self.miou_run.reset()
        self.iou.reset()

    def save(self, epoch, mode, **kwargs):

        if mode == 'best_performance':
            best_miou = kwargs['best_miou']
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'best_miou': best_miou,
                'metrics': self.metrics}, 
                f'./{self.model_path}/best_miou.pth.tar')

        elif mode == 'last':
            best_miou = kwargs['best_miou']
            v_miou = kwargs['v_miou']
            torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(), 
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'best_miou': best_miou,
                        'last_val_miou': v_miou,
                        'metrics': self.metrics,
                        }, 
                        f'./{self.model_path}/last_weights.pth.tar')


    def train(self):
        
        since = time()
        start_epoch = 0
        best_miou = 0.0
        total_per_iou = torch.zeros((self.epochs, 20))
        for epoch in range(1, self.epochs+1) :
            epoch += start_epoch 
            print(f'\n-----------------------\nEpoch {epoch}')
            
            progress = ProgressMeter(len(self.train_loader),
                                    [self.b_t, self.loss_run, self.miou_run],
                                    prefix=f"epoch {epoch} Train")
            self.model.train()
            end = time()

            for iter, batch in enumerate(self.train_loader):            

                inputs = batch['xyz'].to(self.device)
                labels = batch['label'].to(self.device)
                bs = inputs.size(0)

                with torch.set_grad_enabled(True):
                    with torch.autograd.detect_anomaly():
                        outputs, trans, trans_feat = self.model(inputs)
                        self.optimizer.zero_grad()
                        outputs = outputs.view(-1, self.nclasses)
                        labels = labels.view(-1)
                        loss = self.criterion(outputs.cpu(), labels.type(torch.LongTensor).cpu())
                        loss += (self.criterion_feat(trans_feat)*0.001).cpu()
                        loss.backward()
                        self.optimizer.step()

                # mIoU
                self.iou.addBatch(torch.argmax(outputs, 1), labels)
                miou, per_iou = self.iou.getIoU()
                self.miou_run.update(miou, bs)
                total_per_iou[epoch-1] = per_iou

                # statistics
                loss = loss.item()
                self.loss_run.update(loss, bs)
                self.b_t.update(time()-end)
                end = time()

                progress.display(iter)

            t_loss, t_miou= self.loss_run.avg, self.miou_run.avg
            
            self.metrics['t_loss'].append(t_loss)
            self.metrics['t_miou'].append(t_miou)
            
            print('\n[Train] Loss {:.4f} | mIoU {:.4f}'.format(t_loss, t_miou))
            

            # start validation ------------------------------------------------------------------------------------------------------------------------------------------------
            v_loss, v_miou = self.val(epoch)
            self.metrics['v_loss'].append(v_loss)
            self.metrics['v_miou'].append(v_miou)

            self.scheduler.step()
            
            # save history
            with open(f'./{self.model_path}/result.csv', 'a') as epoch_log:
                epoch_log.write('\n\t{} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f}'
                                .format(epoch, t_loss, v_loss, t_miou, v_miou))

            # save model per epochs         --------------------------------------------------
            self.save(epoch, 'last', best_miou=best_miou, v_miou=v_miou)

            # Save best psnr model to file       --------------------------------------------------
            if v_miou > best_miou:
                print('ssim improved from {:.4f} to {:.4f}. epoch {}'.format(best_miou, v_miou, epoch))
                best_miou = v_miou
                self.save(epoch, 'best_performance', best_miou=best_miou)

            # early stopping                --------------------------------------------------
            self.earlystop(val_loss=v_loss, model=self.model, epoch=epoch, optimizer=self.optimizer, best_miou=best_miou, metrics=self.metrics)
            if self.earlystop.early_stop:
                break

            # tensorboard                   --------------------------------------------------
            if self.writer_train != None:
                self.writer_train.add_scalar("Loss", t_loss, epoch)
                self.writer_train.add_scalar("mIoU", t_miou, epoch)

                self.writer_val.add_scalar("Loss", v_loss, epoch)
                self.writer_val.add_scalar("mIoU", v_miou, epoch)
                
                self.writer_train.flush()
                self.writer_train.close()
                self.writer_val.flush()
                self.writer_val.close()

            time_elapsed = time() - since
            print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))


    def val(self, epoch):
        self.reset()
        total_per_iou = torch.zeros((self.epochs, 20))
        progress = ProgressMeter(len(self.val_loader),
                                [self.b_t, self.loss_run, self.miou_run],
                                 prefix=f'epoch {epoch} Test')
        
        self.model.eval()
        with torch.no_grad():
            end = time()
            for iter, batch in enumerate(self.val_loader):

                inputs = batch['xyz'].to(self.device)
                labels = batch['label'].to(self.device)
                bs = inputs.size(0)

                outputs, trans, trans_feat  = self.model(inputs) # rgb 3channel 
                outputs = outputs.view(-1, self.nclasses)
                labels = labels.view(-1)
                loss = self.criterion(outputs.cpu(), labels.type(torch.LongTensor).cpu())
                loss += (self.criterion_feat(trans_feat)*0.001).cpu()

                self.iou.addBatch(torch.argmax(outputs, 1), labels)
                miou, per_iou = self.iou.getIoU()
                self.miou_run.update(miou, bs)
                total_per_iou[epoch-1] = per_iou

                loss = loss.item()
                self.loss_run.update(loss, bs)
                self.b_t.update(time()-end)
                end = time()

                progress.display(iter)


        v_loss, v_miou = self.loss_run.avg, self.miou_run.avg
            
        print('\n[Validation] | Loss {:.4f} | mIoU {:.4f} '.format(v_loss,v_miou))

        return v_loss, v_miou
