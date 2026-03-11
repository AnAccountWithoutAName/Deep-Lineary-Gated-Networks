import torch


class DLGN(torch.nn.modules):
    def __init__(self,input_dim,hidden_dim,num_layers, beta = 1):

        super.__init__()
        self.beta = beta
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.hiddens = [torch.nn.Parameter(torch.randn(input_dim,hidden_dim),requires_grad=True)]
        self.hiddens += [torch.nn.Parameter(torch.randn(hidden_dim,hidden_dim),requires_grad=True) for _ in range(num_layers - 1)]
        self.u1,self.ulast = torch.nn.Parameter(torch.randn(hidden_dim,1),requires_grad=True),torch.nn.Parameter(torch.randn(hidden_dim,1))
        self.Us = [self.u1] + [torch.nn.Parameter(torch.randn(hidden_dim,hidden_dim),requires_grad=True) for _ in range(num_layers - 1)] + [self.ulast]
        self.biases = [torch.nn.Parameter(torch.randn(hidden_dim)) for _ in range(num_layers)]

    def forward(self,x):

        self.V = [self.hiddens[0]]
        h = torch.nn.Sigmoid(self.beta*(self.V[0]@x + self.biases[0])) * self.Us[0]

        for i in range(1,self.num_layers):
            self.V.append(self.V[-1]@self.hiddens[i])
            h = torch.nn.Sigmoid(self.beta*(self.V[i]@x + self.biases[i]))*(self.Us[i]@h)
        return torch.dot(h,self.Us[-1])



        
        
