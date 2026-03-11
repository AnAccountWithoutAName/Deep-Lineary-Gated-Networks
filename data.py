import torch
from torch.utils.data import DataLoader,TensorDataset,random_split
import lightning as L
import random
import matplotlib.pyplot as plt
from torchvision import transforms





def generate_data(block_size, grid_size, num_samples, background_mu = 0, data_mu = 5):


    # create background num_samples times
    grid = torch.normal(torch.full((num_samples,grid_size,grid_size),background_mu,dtype=torch.float32))
    
    # pick a random block for each of the num_samples

    num_blocks = (grid_size//block_size)**2
    block_index = torch.randint(high = num_blocks,size = (num_samples,))
 
    block_xs = (block_index*block_size)%grid_size
    block_ys = (block_index//(block_size-1))*block_size
    add_x, add_y = torch.meshgrid(torch.arange(block_size),torch.arange(block_size))
    block_xs = block_xs.view(-1,1,1)
    block_ys = block_ys.view(-1,1,1)
    id_x,id_y = block_xs + add_x, block_ys + add_y
    batch_id = torch.arange(num_samples).view(-1,1,1).expand(-1,block_size,block_size)

    # create data of block size

    data = torch.normal(torch.full((num_samples,block_size,block_size), data_mu, dtype = torch.float32))
    
    # replace block with data

    grid[batch_id,id_y,id_x] = data



    return grid, data, block_index

 
    



def generate_labels(data,depth = 4):

    data_size = data.size(1)*data.size(2)
    num_samples = data.size(0)

    # initialize weights for each dimension

    weights = torch.eye(data_size,2**depth - 1,dtype = torch.float32)

    # initialize tree indices for each sample

    index = torch.ones(num_samples, dtype = torch.int)

    # flatten and mean center data
    data -= torch.mean(data,dim = (1,2), keepdim= True,dtype = torch.float32)
    data = torch.flatten(data,1,-1)

    print(data)

    # traverse till leaf
    while torch.all(index <= 2**(depth - 1) - 1):
        condition =  (data @ weights[:,index - 1] >= 0).diag()
        index = 2*(index)*(~condition) + (2*index + 1)*condition

    

    # assign labels
    labels = index % 2

    return weights, labels




class ImageDataModule(L.LightningDataModule):
    def __init__(self, block_size, grid_size, num_samples, bg_mu, data_mu, tree_depth, train_size, batch_size, control_group = False):
        super().__init__()
        self.save_hyperparameters()
        self.block_size = block_size
        self.grid_size = grid_size
        self.num_samples = num_samples
        self.bg_mu = bg_mu
        self.data_mu = data_mu
        self.depth = tree_depth
        self.train_size = train_size
        self.batch_size = batch_size
        self.control = control_group




    def setup(self,stage):
        self.grids, self.data, self.block_index = self._generate_data(self.block_size,self.grid_size,self.num_samples,self.bg_mu,self.data_mu)
        self.weights, self.labels = self._generate_labels(self.data,self.depth)
        # convert to tensor dataset
     

        if self.control:

      

            self.grids = self.grids.repeat(1,3,1,1)
            preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
               # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

           # self.grids = preprocess(self.grids)
            


        

        self.full_dataset = TensorDataset(self.grids,self.labels)




        # split into train and validation sets
        # add generator later

        self.train,self.val = random_split(self.full_dataset,[self.train_size, 1 - self.train_size])
    
    def train_dataloader(self):
        return DataLoader(self.train, batch_size = self.batch_size)
    def val_dataloader(self):
        return DataLoader(self.val,batch_size = self.batch_size )
        


    def _generate_data(self,block_size, grid_size, num_samples, background_mu = 0, data_mu = 5):


        # create background num_samples times
        grid = torch.normal(torch.full((num_samples,grid_size,grid_size),background_mu,dtype=torch.float32))
        
        # pick a random block for each of the num_samples

        num_blocks = (grid_size//block_size)**2
        block_index = torch.randint(high = num_blocks,size = (num_samples,))
    
        block_xs = (block_index*block_size)%grid_size
        block_ys = (block_index//(block_size-1))*block_size
        add_x, add_y = torch.meshgrid(torch.arange(block_size),torch.arange(block_size))
        block_xs = block_xs.view(-1,1,1)
        block_ys = block_ys.view(-1,1,1)
        id_x,id_y = block_xs + add_x, block_ys + add_y
        batch_id = torch.arange(num_samples).view(-1,1,1).expand(-1,block_size,block_size)

        # create data of block size

        data = torch.normal(torch.full((num_samples,block_size,block_size), data_mu, dtype = torch.float32))
        
        # replace block with data

        grid[batch_id,id_y,id_x] = data



        return grid.unsqueeze(1), data, block_index
    def _generate_labels(self,data,depth = 4):

        data_size = data.size(1)*data.size(2)
        num_samples = data.size(0)

        # initialize weights for each dimension

        weights = torch.eye(data_size,2**depth - 1,dtype = torch.float32)

        # initialize tree indices for each sample

        index = torch.ones(num_samples, dtype = torch.int)

        # flatten and mean center data
        data -= torch.mean(data,dim = (1,2), keepdim= True,dtype = torch.float32)
        data = torch.flatten(data,1,-1)



        # traverse till leaf
        while torch.all(index <= 2**(depth - 1) - 1):
            condition =  (data @ weights[:,index - 1] >= 0).diag()
            index = 2*(index)*(~condition) + (2*index + 1)*condition

        

        # assign labels
        labels = index % 2

        return weights, labels






# def set_npseed(seed):
#     np.random.seed(seed)


# def set_torchseed(seed):
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False


# #classification data

# class NumClassError(Exception):
#     def __init__(self, message, error_code=None):
#         self.message = message
#         self.error_code = error_code
#         super().__init__(self.message)

#     def __str__(self):
#         if self.error_code:
#             return f"Error Code {self.error_code}: {self.message}"
#         return self.message



# import torch

# def data_gen_decision_tree(num_data=1000, dim=2, seed=0, w_list=None, b_list=None, vals=None, num_levels=2, threshold=0, num_classes=2, w_init='random'):
#     torch.manual_seed(seed)

#     num_internal_nodes = 2**num_levels - 1
#     num_leaf_nodes = 2**num_levels
    
#     # 1. Initialize Labels (vals)
#     if vals is None:
#         vals = torch.arange(0, num_internal_nodes + num_leaf_nodes, 1, dtype=torch.int32) % num_classes
#         vals[:num_internal_nodes] = -99 

#     # 2. Initialize Weights (w_list)
#     if w_list is None:
#         if w_init == 'random':
#             w_list = torch.randn((num_internal_nodes, dim))
#             w_list = w_list / torch.norm(w_list, dim=1, keepdim=True)
#         elif w_init == 'axes':
#             w_list = torch.eye(num_internal_nodes, dim) if num_internal_nodes <= dim else torch.eye(dim).repeat(num_internal_nodes//dim + 1, 1)[:num_internal_nodes]

#     if b_list is None:
#         b_list = torch.zeros((num_internal_nodes))

#     # 3. Generate and Prepare Data
#     data_x = torch.randn((num_data, dim))
#     data_x /= torch.norm(data_x, dim=1, keepdim=True)
    
#     # 4. Vectorized Tree Traversal (Your Logic Integrated)
#     # We use 0-based indexing: root is 0, children are (2i+1) and (2i+2)
#     curr_index = torch.zeros(num_data, dtype=torch.long)
    
#     # Pre-calculate all wx + b scores for the traversal
#     relevant_stats = data_x @ w_list.T + b_list

#     for _ in range(num_levels):
#         # Determine if we go right (score > 0) or left (score <= 0)
#         # gather picks the relevant wx+b score for each sample's current node
#         decision_variable = torch.gather(relevant_stats, 1, curr_index.unsqueeze(1)).squeeze(1)
        
#         condition = (decision_variable > 0).long()
#         # 0-based heap indexing: Left = 2i + 1, Right = 2i + 2
#         curr_index = 2 * curr_index + 1 + condition

#     # 5. Labeling and Pruning
#     labels = vals[curr_index]
    
#     # Thresholding: find min distance to any hyperplane to ensure data isn't "on the line"
#     bound_dist, _ = torch.min(torch.abs(relevant_stats), dim=1)
#     mask = bound_dist > threshold
    
#     data_x_pruned = data_x[mask]
#     labels_pruned = labels[mask]
    
#     # 6. Final Node Activity Stats
#     # We track which internal/leaf nodes are "active" for the pruned dataset
#     nodes_active = torch.zeros((len(data_x_pruned), num_internal_nodes + num_leaf_nodes), dtype=torch.int32)
#     relevant_stats_pruned = torch.sign(data_x_pruned @ w_list.T + b_list)
    
#     for node in range(num_internal_nodes + num_leaf_nodes):
#         if node == 0:
#             nodes_active[:, 0] = 1
#             continue
        
#         parent = (node - 1) // 2
#         is_right_child = (node % 2 == 0)
        
#         if is_right_child:
#             nodes_active[:, node] = nodes_active[:, parent] * (relevant_stats_pruned[:, parent] > 0).int()
#         else:
#             nodes_active[:, node] = nodes_active[:, parent] * (relevant_stats_pruned[:, parent] <= 0).int()

#     stats = nodes_active.sum(dim=0).float()

#     return ((data_x_pruned, labels_pruned), (w_list, b_list, vals), stats)
# def generate_synthetic_data(param_dct, w_init):
#     # print()
#     num_data = param_dct['num_data']
#     seed = param_dct['seed']
#     threshold = param_dct['threshold']
#     num_levels = param_dct['num_levels']
#     dim = param_dct['dim']
#     num_classes = param_dct['num_classes']

#     if num_data < dim:
#         print('num_data is less than dim. Needs to be >=. returning...')
#         return

#     ((data_x, labels), (w_list, b_list, vals), stats) = data_gen_decision_tree(
#                                             dim=dim, seed=seed, num_levels=num_levels,
#                                             num_data=num_data, threshold= threshold, num_classes= num_classes,
#                                             w_init=w_init)
#     seed_set=seed
#     w_list_old = np.array(w_list)
#     b_list_old = np.array(b_list)
#     # print(sum(labels==1))
#     # print(sum(labels==0))
#     # print("Seed= ",seed_set)
#     num_data = len(data_x)
#     num_train= num_data//2
#     num_vali = num_data//4
#     num_test = num_data//4
#     train_data = data_x[:num_train,:]
#     train_data_labels = labels[:num_train]

#     vali_data = data_x[num_train:num_train+num_vali,:]
#     vali_data_labels = labels[num_train:num_train+num_vali]

#     test_data = data_x[num_train+num_vali :,:]
#     test_data_labels = labels[num_train+num_vali :]

#     return [(w_list_old, b_list_old, dim), (train_data, train_data_labels, num_train),
#             (vali_data, vali_data_labels, num_vali), (test_data, test_data_labels, num_test), (data_x, labels)]





    


    



if __name__ == "__main__":
    dm = ImageDataModule(block_size = 3,
                         grid_size = 6,
                         num_samples = 100,
                         bg_mu = 0,
                         data_mu = 5,
                         tree_depth = 4,
                         train_size = 0.8,
                         batch_size = 32,
                         control_group= True
                         )
    dm.setup(stage = 'fit')

    train_loader = dm.train_dataloader()

    mini_batch = list(train_loader)[0]

    x,y = mini_batch

    plt.imshow(x[0,0])
    
    plt.show()

    plt.imshow(x[0,1])

    plt.show()


   
    
    


 
    




  

 
  


 
   