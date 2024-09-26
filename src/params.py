def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    on.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def save_model_weights(model,filename, verbose=1,cp_folder=""):
    
    if verbose: 
        print(f"Saving weights to{os.path.join(cp_folder,filename)}/n")
    torch.save(model.state_dict(),os.path.join(cp_folder,filename))
    
def count_parameters(model,all=False):
    if all: 
        return sum(p.numel() for p in model.parameters())
    else: 
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
#This part calculates only the total number of trainable parameters 
#(those where requires_grad=True), which are the ones updated during backpropagation.

#Usage:
#Use all=True if you want to get the total parameter count.
#Use all=False if you want to get the count of only the trainable parameters.
