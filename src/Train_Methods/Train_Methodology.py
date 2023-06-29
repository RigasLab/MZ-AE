import torch
import torch.nn as nn
import matplotlib.pyplot as plt




class Train_Methodology():

    def time_evolution(self, initial_x_n, initial_x_seq, initial_Phi_n, ph_size):

        """
        Calculates multistep prediction from koopman and seqmodel while training
        Inputs
        ------
        initial_x_n (torch tensor): [bs obsdim]
        initial_x_seq (torch tensor): [bs seq_len obsdim]
        initial_Phi_n (torch tensor): [bs statedim]
        ph_size (int) : variable pred_horizon acccording to future data available

        Returns
        -------
        x_nn_hat_ph (torch_tensor): [bs pred_horizon obsdim]
        Phi_nn_hat (torch_tensor): [bs pred_horizon statedim]
        """

        x_n   = initial_x_n 
        x_seq = initial_x_seq
        koop_out_ph = x_n.clone()[:,None,...]   #[bs 1 obsdim]
        if not self.deactivate_seqmodel:
            seqmodel_out_ph = x_n.clone()[:,None,...]   #[bs 1 obsdim]
        x_nn_hat_ph = x_n.clone()[:,None,...]   #[bs 1 obsdim]
        Phi_nn_hat_ph = initial_Phi_n.clone()[:,None,...] #[bs 1 statedim]

        #Evolving in Time
        for t in range(ph_size):
            
            #collecting koopman prediction
            koop_out = self.model.koopman(x_n)
            if t == 0:
                koop_out_ph[:,0,...] = koop_out
            else:
                koop_out_ph = torch.cat((koop_out_ph, koop_out[:, None, ...]), 1)

            if self.deactivate_seqmodel:
                x_nn_hat = koop_out
             #collecting seqmodel prediction
            else:
                seqmodel_out = self.model.seqmodel(x_seq)
                if t == 0:                      
                    seqmodel_out_ph[:,0,...] = seqmodel_out
                else:
                    seqmodel_out_ph = torch.cat((seqmodel_out_ph, seqmodel_out[:, None, ...]), 1)
                
                x_nn_hat = koop_out + seqmodel_out 
            
            Phi_nn_hat = self.model.autoencoder.recover(x_nn_hat)
            

            #concatenating prediction
            x_nn_hat_ph   = torch.cat((x_nn_hat_ph,x_nn_hat[:,None,...]), 1)
            Phi_nn_hat_ph = torch.cat((Phi_nn_hat_ph,Phi_nn_hat[:,None,...]), 1)

            #forming sequence for next step prediction
            # if t != self.pred_horizon-1 : 
            x_seq = torch.cat((x_seq[:,1:,...],x_n[:,None,...]), 1)
            x_n = x_nn_hat
        
        if self.deactivate_seqmodel:
            return x_nn_hat_ph[:,1:,...], Phi_nn_hat_ph[:,1:,...], koop_out_ph
        else:
            return x_nn_hat_ph[:,1:,...], Phi_nn_hat_ph[:,1:,...], koop_out_ph, seqmodel_out_ph
        




    def train_test_loss(self, mode = "Train", dataloader = None):
        '''
        One Step Prediction method
        Requires: dataloader, model, optimizer
        '''

        if mode == "Train":
            dataloader = self.train_dataloader 
            self.model.train() 
        elif mode == "Test":
            dataloader = self.test_dataloader if dataloader != None else dataloader
            self.model.eval()
        else:
            print("mode can be Train or Test")
            return None

        num_batches = len(dataloader)
        total_loss, total_ObsEvo_Loss, total_Autoencoder_Loss, total_StateEvo_Loss,\
            total_KoopEvo_Loss, total_Residual_Loss  = 0,0,0,0,0,0
        total_koop_ptg, total_seqmodel_ptg = 0,0
        

        for Phi_seq, Phi_nn_ph in dataloader:
            
            ph_size = Phi_nn_ph.shape[1] # pred_horizon size can vary depending on future steps available in data

            Phi_n   = torch.squeeze(Phi_seq[:,-1,...])  
            Phi_n = Phi_n[None,...] if (Phi_n.ndim == self.state_ndim) else Phi_n #[bs statedim]
            Phi_n_ph = torch.cat((Phi_n[:,None,...], Phi_nn_ph[:,:-1,...]), 1)    #[bs ph_size statedim]
            
            #flattening batchsize seqlen / batchsize pred_horizon
            Phi_seq = torch.flatten(Phi_seq, start_dim = 0, end_dim = 1) #[bs*seqlen, statedim]
            Phi_nn_ph  = torch.flatten(Phi_nn_ph, start_dim = 0, end_dim = 1) #[bs*ph_size, statedim]
            #obtain observables
            x_seq, Phi_seq_hat = self.model.autoencoder(Phi_seq)
            x_nn_ph , Phi_nn_hat_ph_nolatentevol = self.model.autoencoder(Phi_nn_ph)

            #reshaping tensors in desired form
            sd = (self.statedim,) if str(type(self.statedim)) == "<class 'int'>" else self.statedim
            
            Phi_nn_ph   = Phi_nn_ph.reshape(int(Phi_nn_ph.shape[0]/ph_size), ph_size, *sd) #[bs ph_size statedim]
            Phi_nn_hat_ph_nolatentevol = Phi_nn_hat_ph_nolatentevol.reshape(int(Phi_nn_hat_ph_nolatentevol.shape[0]/ph_size), ph_size, *sd) #[bs pred_horizon statedim]
            Phi_seq_hat = Phi_seq_hat.reshape(int(Phi_seq_hat.shape[0]/self.seq_len), self.seq_len, *sd) #[bs seqlen statedim]
            Phi_n_hat   = torch.squeeze(Phi_seq_hat[:, -1, :])
            Phi_n_hat = Phi_n_hat[None,...] if (Phi_n_hat.ndim == self.state_ndim) else Phi_n_hat #[bs statedim]

            Phi_n_hat_ph = torch.cat((Phi_n_hat[:,None,...], Phi_nn_hat_ph_nolatentevol[:,:-1,...]), 1)  #obtaining decoded state tensor
             
            x_nn_ph  = x_nn_ph.reshape(int(x_nn_ph.shape[0]/ph_size), ph_size, self.num_obs) #[bs ph_size obsdim]
            x_seq = x_seq.reshape(int(x_seq.shape[0]/self.seq_len), self.seq_len, self.num_obs) #[bs seqlen obsdim]
            x_n   = torch.squeeze(x_seq[:,-1,:])   
            x_n   = x_n[None,...] if (x_n.ndim == 1) else x_n #[bs obsdim]
            x_seq = x_seq[:,:-1,:] #removing the current timestep from sequence. The sequence length is one less than input
            
            # #Evolving in Time
            if self.deactivate_seqmodel:
                x_nn_hat_ph, Phi_nn_hat_ph, koop_nn_ph = self.time_evolution(x_n, x_seq, Phi_n, ph_size)

            else:
                x_nn_hat_ph, Phi_nn_hat_ph, koop_nn_ph, seqmodel_nn_ph = self.time_evolution(x_n, x_seq, Phi_n, ph_size)

            #calculating residual
            residual = x_nn_ph - koop_nn_ph

            #Calculating loss
            mseLoss      = nn.MSELoss()
            KoopEvo_Loss = mseLoss(koop_nn_ph, x_nn_ph)
            if not self.deactivate_seqmodel:
                Residual_Loss = mseLoss(seqmodel_nn_ph, residual)
            Autoencoder_Loss = mseLoss(Phi_n_hat_ph, Phi_n_ph)
            StateEvo_Loss = mseLoss(Phi_nn_hat_ph, Phi_nn_ph)

            #calculating l1 norm of the matrix
            # kMatrix = self.model.koopman.getKoopmanMatrix(requires_grad = False)
            # l1_norm = torch.norm(kMatrix, p=1)
            # seqnorm = torch.norm(seqmodel_nn_ph, p = 'fro')**2
            if not self.deactivate_seqmodel:
                loss = (KoopEvo_Loss + Residual_Loss) + \
                        100*(Autoencoder_Loss) #+ StateEvo_Loss #+ self.seq_model_weight*seqnorm
            else:
                loss = 0.1*(KoopEvo_Loss) + \
                        100*(Autoencoder_Loss) #+ StateEvo_Loss #+ 0.00001*(torch.norm(abs(Phi_n_hat - Phi_n), float('inf')) + torch.norm(abs(Phi_nn_hat - Phi_nn), float('inf')))#+ 0.1*torch.mean(torch.abs(self.model.koopman.kMatrixDiag)) + 0.1*torch.mean(torch.abs(self.model.koopman.kMatrixUT))#(1e-9)*l1_norm

            if mode == "Train":
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()


            total_loss += loss.item()
            total_KoopEvo_Loss +=  KoopEvo_Loss.item()
            if not self.deactivate_seqmodel:
                total_Residual_Loss += Residual_Loss.item()
            total_Autoencoder_Loss += Autoencoder_Loss.item()
            total_StateEvo_Loss += StateEvo_Loss.item()
            total_koop_ptg         += 0#koop_ptg
            total_seqmodel_ptg     += 0#seq_ptg


        avg_loss             = total_loss / num_batches
        # avg_ObsEvo_Loss      = total_ObsEvo_Loss / num_batches
        avg_KoopEvo_Loss     = total_KoopEvo_Loss / num_batches
        if not self.deactivate_seqmodel:
            avg_Residual_Loss    = total_Residual_Loss / num_batches
        avg_Autoencoder_Loss = total_Autoencoder_Loss / num_batches
        avg_StateEvo_Loss    = total_StateEvo_Loss / num_batches
        avg_koop_ptg         = total_koop_ptg / num_batches
        avg_seqmodel_ptg     = total_seqmodel_ptg / num_batches

        Ldict = {'avg_loss': avg_loss, 'avg_KoopEvo_Loss': avg_KoopEvo_Loss, 'avg_Residual_Loss': 0, \
                 'avg_Autoencoder_Loss': avg_Autoencoder_Loss, 'avg_StateEvo_Loss': avg_StateEvo_Loss,\
                 'avg_koop_ptg': avg_koop_ptg, 'avg_seqmodel_ptg': avg_seqmodel_ptg} 
        if not self.deactivate_seqmodel:
            Ldict['avg_Residual_Loss'] = avg_Residual_Loss

        return Ldict
    def training_loop(self):
        '''
        Requires:
        model, optimizer, train_dataloader, val_dataloader, device
        '''
        print("Device: ", self.device)
        print("Untrained Test\n--------")
        test_Ldict = self.train_test_loss("Test", self.test_dataloader)
        # test_loss, test_ObsEvo_Loss, test_Autoencoder_Loss, test_StateEvo_Loss, test_koop_ptg, test_seqmodel_ptg = 
        print(f"Test Loss: {test_Ldict['avg_loss']}, KoopEvo : {test_Ldict['avg_KoopEvo_Loss']}, Residual : {test_Ldict['avg_Residual_Loss']}, Auto : {test_Ldict['avg_Autoencoder_Loss']}, StateEvo : {test_Ldict['avg_StateEvo_Loss']}")

        # min train loss
        self.min_train_loss = 1000 
        
        for ix_epoch in range(self.load_epoch, self.load_epoch + self.nepochs):

            #learning rate customization
            if not self.deactivate_lrscheduler:
                before_lr = self.optimizer.param_groups[0]["lr"]
                self.scheduler.step()
                after_lr = self.optimizer.param_groups[0]["lr"]
                print("Epoch %d: SGD lr %.6f -> %.6f" % (ix_epoch, before_lr, after_lr))
            
            #Activating seq_model in between if asked
            if self.nepoch_actseqmodel!=0:
                if ix_epoch == self.nepoch_actseqmodel:
                    self.deactivate_seqmodel = False
                    for param in self.model.seqmodel.parameters():
                        param.requires_grad = True
                    
                    print("SEQMODEL : ", not self.deactivate_seqmodel)

            # train_loss, train_ObsEvo_Loss, train_Autoencoder_Loss, train_StateEvo_Loss, train_koop_ptg, train_seqmodel_ptg = 
            train_Ldict = self.train_test_loss("Train")
            test_Ldict  = self.train_test_loss("Test", self.test_dataloader)
            # test_loss, test_ObsEvo_Loss, test_Autoencoder_Loss, test_StateEvo_Loss, test_koop_ptg, test_seqmodel_ptg  = 
            
            #printing and saving data
            print(f"Epoch {ix_epoch} ")
            print(f"Train Loss: {train_Ldict['avg_loss']}, KoopEvo : {train_Ldict['avg_KoopEvo_Loss']}, Residual : {train_Ldict['avg_Residual_Loss']}, Auto : {train_Ldict['avg_Autoencoder_Loss']}, StateEvo : {train_Ldict['avg_StateEvo_Loss']}")
            self.log.writerow({"epoch":ix_epoch,"Train_Loss":train_Ldict['avg_loss'], "Train_KoopEvo_Loss":train_Ldict['avg_KoopEvo_Loss'], "Train_Residual_Loss":train_Ldict['avg_Residual_Loss'], "Train_Autoencoder_Loss":train_Ldict["avg_Autoencoder_Loss"], "Train_StateEvo_Loss":train_Ldict["avg_StateEvo_Loss"],\
                                                "Test_Loss":test_Ldict['avg_loss'], "Test_KoopEvo_Loss":test_Ldict['avg_KoopEvo_Loss'], "Test_Residual_Loss":test_Ldict['avg_Residual_Loss'],  "Test_Autoencoder_Loss":test_Ldict["avg_Autoencoder_Loss"], "Test_StateEvo_Loss":test_Ldict["avg_StateEvo_Loss"],\
                                                "Train_koop_ptg": 0, "Train_seqmodel_ptg": 0,\
                                                "Test_koop_ptg": 0, "Test_seqmodel_ptg": 0})
            self.logf.flush()
            
            if self.min_train_loss > train_Ldict["avg_loss"]:
                self.min_train_loss = train_Ldict["avg_loss"]
                torch.save(self.model.state_dict(), self.exp_dir+'/'+ self.exp_name+"/model_weights/min_train_loss".format(epoch=ix_epoch))

            if (ix_epoch%self.nsave == 0):
                #saving weights
                torch.save(self.model.state_dict(), self.exp_dir+'/'+ self.exp_name+"/model_weights/at_epoch{epoch}".format(epoch=ix_epoch))
        
        #saving weights
        torch.save(self.model.state_dict(), self.exp_dir+'/'+ self.exp_name+"/model_weights/at_epoch{epoch}".format(epoch=ix_epoch))
        # writer.close()
        self.logf.close()