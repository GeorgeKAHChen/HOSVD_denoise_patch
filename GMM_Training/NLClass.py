#=============================================================================
#
#       Group Attribute Random Walk Program
#       NLRWClass.py
#
#       Copyright by KazukiAmakawa, all right reserved
#
#=======================================================
#
#       Intro
#       This is the Class file we build for our new random walk classification
#       You can use this class directly. The parameter usage is same as linear
#       Classification
#
#=============================================================================
import torch
import torch.nn as nn


class GMMDense(nn.Module):
    def __init__(self, 
                input_features, 
                output_features, 
                log_mark = False, 
                device = "cuda"):

        super(GMMDense, self).__init__()
        self.input_features  = input_features
        self.output_features = output_features
        self.device          = device
        self.log_mark        = log_mark

        self.prob            = nn.Parameter(torch.Tensor(output_features, 1)).to(self.device)
        self.Mu              = nn.Parameter(torch.Tensor(output_features, input_features)).to(self.device)
        self.Sigma           = nn.Parameter(torch.Tensor(output_features, input_features, input_features)).to(self.device)
        
        self.prob.data.uniform_(0, 1)
        self.Mu.data.uniform_(0, 1)
        self.Sigma.data.uniform_(0.5, 2)


    def Unit_prob(self):
        Zero      = torch.zeros(self.output_features, 1).to(self.device)
        self.prob = nn.Parameter(torch.max(self.prob, Zero))
        sum_prob  = torch.sum(self.prob).to(self.device)
        self.prob = nn.Parameter(self.prob / sum_prob)
        Zero      = torch.zeros(self.Sigma.size()).to(self.device)
        self.Sigma = nn.Parameter(torch.max(self.Sigma, Zero))



    def Sigma_Cal(self):
        Zero = torch.zeros(1).to(self.device)
        epsilon = torch.Tensor([1e-5]).to(self.device)

        sigma_inv = torch.zeros([self.output_features, self.input_features, self.input_features]).to(self.device)
        sigma_det = torch.zeros(self.output_features).to(self.device)
        for i in range(0, len(self.Sigma)):
            sigma_inv[i] = torch.inverse(self.Sigma[i])
            det = torch.sqrt(torch.max(torch.det(self.Sigma[i]), Zero)).to(self.device)
            if torch.gt(det, epsilon):
                if not self.log_mark:
                    sigma_det[i] = 0.3989422804014327/det
                else:
                    sigma_det[i] = 6.283185307179586 * det

            else:
                if not self.log_mark:
                    sigma_det[i] = 0
                else:
                    sigma_det[i] = epsilon
                

        return sigma_inv, sigma_det



    def Output_Model(self):
        import numpy as np
        import math
        inp_fea     = self.input_features
        oup_fea     = self.output_features
        prob        = self.prob.cpu().detach().numpy()
        Mu          = self.Mu.cpu().detach().numpy()
        Sigma       = self.Sigma.cpu().detach().numpy()
        OupArr      = str(inp_fea) + "\t" + str(oup_fea) + "\n"
        inv, det    = self.Sigma_Cal(self)
        det         = self.det.cpu().detach().numpy()
        true_weight = math.log(6.283185307179586 * det)
        #print(np.shape(true_weight))

        for i in range(0, len(prob)):
            OupArr += str(true_weight[i] / prob[i][0])
            OupArr += "\n"
            for j in range(0, len(Mu[i])):
                OupArr += str(Mu[i][j])
                OupArr += "\t"
            OupArr += "\n"
            for j in range(0, len(Sigma[i])):
                for k in range(0, len(Sigma[i][j])):
                    OupArr += str(Sigma[i][j][k])
                    OupArr += "\t"
                OupArr += "\n"
        
        FileName = "Output/Model" + str(inp_fea) + ".out"
        File = open(FileName, "w")
        File.write(OupArr)
        File.close()
        print("Model Saved Succeed, file name: Model" + str(inp_fea) + ".out")
        return


    def forward(self, input):
        import math
        Zero = torch.zeros(1).to(self.device)
                                            # def para =: 0 as tensor
        epsilon = torch.Tensor([1e-5]).to(self.device)
        GMMDense.Unit_prob(self)            # Initial self.prob parameter as normal
        
        if len(input) == 1:
            GMMDense.Output_Model(self)     # Save Model
            return 

        sigma_inv, sigma_det = GMMDense.Sigma_Cal(self)
                                            # cal sigma-inverse and sigma-det to calculate
        outputs = torch.zeros(self.output_features, len(input)).to(self.device)

        if not self.log_mark:
            for j in range(0, self.output_features):
                x_neg_mu = input - self.Mu[j]
                #outputs[j] = sigma_det[j] * torch.exp(-0.5 * (
                outputs[j] = torch.exp(-0.5 * (
                    torch.max( torch.sum(torch.mul(torch.mm(x_neg_mu, sigma_inv[j]), x_neg_mu), dim = 1), Zero)
                    ))
            outputs = outputs.t()
            SumOps = torch.max(torch.sum(outputs, dim = 1), epsilon)
            SumOps = torch.reshape(SumOps, [-1, 1])
            outputs /= SumOps
            #print(outputs)
            return outputs
        
        else:
            for j in range(0, self.output_features):
                x_neg_mu = input - self.Mu[j]
                outputs[j] = -0.5 * (torch.sum(  torch.mul(torch.mm(x_neg_mu, sigma_inv[j]), x_neg_mu), dim = 1) + torch.log(sigma_det[j])) + torch.log(self.prob[j])
            outputs = outputs.t()
            #print(outputs.size())
            #SumOps = torch.max(torch.sum(outputs, dim = 1), epsilon)
            #SumOps = torch.reshape(SumOps, [-1, 1])
            #outputs /= SumOps
            #print(outputs)
            return outputs



    def extra_repr(self):
        #Output the io size for visible
        return 'in_features={}, out_features={}'.format(
            self.input_features, self.output_features is not None
        )
        