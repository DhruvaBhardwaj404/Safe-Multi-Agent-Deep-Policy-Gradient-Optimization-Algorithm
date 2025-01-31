import torch

def one_hot_encode(action_prob: torch.tensor, action_size:int):
    """
    Convert n-dim tensor of probabilities of action to one hot encoded tensor.
    @param action_prob: vector of predicted probabilities from policy of an agent
    @type action_prob: n-dim tensor
    @param action_size: number of actions
    @type action_size: int
    @return: one hot-encoded vector
    @rtype: n-dim tensor
    """

    max_act_ind = torch.argmax(action_prob)
    ohe_vector = [0 for _ in range(0,action_size)]
    ohe_vector[max_act_ind] = 1
    return ohe_vector

def convert_dict_to_tensors(data:dict):
    """
    Use to flatten a dict to 2D tensor
    @param data: dictionary or TensorDict
    @type data: dict/ TensorDict
    @return: 2D tensor
    @rtype: torch.tensor
    """
    pdata = []

    for k in data:
        pdata.append(data[k])

    return torch.tensor(pdata)

def get_max_action_index(data:torch.tensor):
    act_ind = []
    for d in data:
        act_ind.append(torch.argmax(d).to("cpu").numpy())
    return act_ind