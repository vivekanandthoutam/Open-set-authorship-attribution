import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import time
import datetime
import random
import gc
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, AdamW
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data import TensorDataset, random_split
from numba import cuda
from GPUtil import showUtilization as gpu_usage
import os
from test import test_function

if torch.cuda.is_available():     
    device = torch.device("cuda")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    n_gpu=torch.cuda.device_count()

    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

dataset=pd.read_csv("./Data/train.csv")
dataset_filtered=dataset.loc[:,['real_label','review']]




labels = dataset_filtered.real_label.values
sentences=dataset_filtered.review.values


output_dir = '/home/vkesana/model_save/'
#output_dir = '/home/vkesana/model_save_closed_set_4_eph/'
#output_dir= '/home/vkesana/model_save_open_set_cls4_opn6_eph/'

tokenizer = AutoTokenizer.from_pretrained(output_dir+'tokenizer/')
model = AutoModelForSequenceClassification.from_pretrained(output_dir)
model.to(device)

"""tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
config = AutoConfig.from_pretrained('bert-base-uncased',num_labels=101,hidden_dropout_prob=0.15)"""

input_ids = []
attention_masks = []

for s in sentences:
    encoded_dict = tokenizer.encode_plus(
                        s,                      # Sentence to encode.
                        truncation=True,
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        max_length = 512,           # Pad & truncate all sentences.
                        padding='max_length',
                        return_attention_mask = True,   # Construct attn. masks.
                        return_tensors = 'pt',     # Return pytorch tensors.
                        return_overflowing_tokens=False,
                   )
    
    # Add the encoded sentence to the list.    
    input_ids.append(encoded_dict['input_ids'])
    
    # And its attention mask (simply differentiates padding from non-padding).
    attention_masks.append(encoded_dict['attention_mask'])

# Convert the lists into tensors.
input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
labels = torch.tensor(labels)



# Combine the training inputs into a TensorDataset.
dataset = TensorDataset(input_ids, attention_masks, labels)

# Create a 90-10 train-validation split.

# Calculate the number of samples to include in each set.
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size

generator = torch.Generator()
generator.manual_seed(44)

# Divide the dataset by randomly selecting samples.
train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

print('{:>5,} training samples'.format(train_size))
print('{:>5,} validation samples'.format(val_size))


batch_size = 16

train_dataloader = DataLoader(
            train_dataset,  # The training samples.
            sampler = RandomSampler(train_dataset), # Select batches randomly
            batch_size = batch_size, # Trains with this batch size.
            num_workers=1
        )


validation_dataloader = DataLoader(
            val_dataset, # The validation samples.
            sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
            batch_size = batch_size, # Evaluate with this batch size.
            num_workers=1)

"""model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', config=config)

model.to(device)"""

optimizer = AdamW(model.parameters(),lr = 5e-5, eps = 1e-8)


epochs = 10
# Total number of training steps is [number of batches] x [number of epochs]. 
# (Note that this is not the same as the number of training samples).
#total_steps = len(train_dataloader) * epochs
gradient_accumulation_steps=1
n_gpu=1
#total_steps= ((len(train_dataloader) // (batch_size * max(1, n_gpu)))// gradient_accumulation_steps* float(epochs))
total_steps = len(train_dataloader) * epochs

# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(optimizer,  num_warmup_steps = 0, num_training_steps = total_steps)

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))

seed_val = 44

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)





training_stats = []

# Measure the total training time for the whole run.
total_t0 = time.time()

# For each epoch...
for epoch_i in range(0, epochs):
    
    # ========================================
    #               Training
    # ========================================
    
    # Perform one full pass over the training set.

    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')

    # Measure how long the training epoch takes.
    t0 = time.time()

    # Reset the total loss for this epoch.
    total_train_loss = 0

    # Put the model into training mode. Don't be mislead--the call to 
    # `train` just changes the *mode*, it doesn't *perform* the training.
    # `dropout` and `batchnorm` layers behave differently during training
    # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
    model.train()

    # For each batch of training data...
    for step, batch in enumerate(train_dataloader):

        # Progress update every 40 batches.
        if step % 40 == 0 and not step == 0:
            # Calculate elapsed time in minutes.
            elapsed = format_time(time.time() - t0)
            
            # Report progress.
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

        # Unpack this training batch from our dataloader. 
        #
        # As we unpack the batch, we'll also copy each tensor to the GPU using the 
        # `to` method.
        #
        # `batch` contains three pytorch tensors:
        #   [0]: input ids 
        #   [1]: attention masks
        #   [2]: labels 
        b_input_ids = batch[0]
        b_input_mask = batch[1]
        b_labels = batch[2]
      #  print(b_input_ids)
      #  print(b_input_mask)
      #  print(b_labels)
        arr_ind=[i for i in range(len(b_labels))]
      #  print(arr_ind)
        counter = 0
        ########
        b_lables_pair_l2 = []
        while(1):
          ind1=np.random.choice(arr_ind)
          ind2=np.random.choice(arr_ind)
          counter =counter + 1
          if (b_labels[ind1].item()!=b_labels[ind2].item()) and ([ind1,ind2] not in b_lables_pair_l2):
         #    print(ind1,ind2)
             #print(b_labels[ind1].item(),b_labels[ind2].item())
             #arr_ind.remove(ind1)
             #arr_ind.remove(ind2)
             b_lables_pair_l2.append([ind1,ind2])
          if(counter>100 or len(b_lables_pair_l2)>=16):
         #   print(arr_ind)
            break
        #print("#######################################################")
        #print(b_lables_pair_l2)

        b_input_ids_l1 = torch.cat(tuple([batch[0][ind:ind+1] for ind in arr_ind])).to(device)
        b_input_mask_l1 = torch.cat(tuple([batch[1][ind:ind+1] for ind in arr_ind])).to(device)
        b_labels_l1 = torch.cat(tuple([batch[2][ind:ind+1] for ind in arr_ind])).to(device)
        # print(b_input_ids)
        # print(b_input_ids_l1)

        # print("******")
        # print(batch[0][0:1])
        # print(batch[0][1:2])
        # sum_pair = np.ceil((batch[0][0:1]+batch[0][1:2])/2)
        # avg_pair=sum_pair.type(torch.int64)
        # print(avg_pair)
        b_input_ids_l2=[]
        b_input_mask_l2=[]
        b_labels_l2=[]

        for ind_pair in b_lables_pair_l2:
            #print("#######################################################")
            #print(ind_pair)
            i=ind_pair[0]
            j=ind_pair[1]
            
            sum_pair=np.ceil((batch[0][i:i+1]+batch[0][j:j+1])/2)
            avg_pair=sum_pair.type(torch.int64)
            b_input_ids_l2.append(avg_pair)
            
            sum_pair_mask=np.ceil((batch[1][i:i+1]+batch[1][j:j+1])/2)
            avg_pair_mask=sum_pair_mask.type(torch.int64)
            
            b_input_mask_l2.append(avg_pair_mask)
            
            b_labels_l2.append(torch.tensor([100]))
            
            #print("#######################################################")
            #print(b_labels_l2)
          
        #print("#######################################################")
        #print(b_labels_l2)

        b_labels_l2 = torch.cat(tuple(b_labels_l2)).to(device)
        b_input_mask_l2 = torch.cat(tuple(b_input_mask_l2)).to(device)
        b_input_ids_l2 = torch.cat(tuple(b_input_ids_l2)).to(device)
        

       

        # Always clear any previously calculated gradients before performing a
        # backward pass. PyTorch doesn't do this automatically because 
        # accumulating the gradients is "convenient while training RNNs". 
        # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
        model.zero_grad()        

        # Perform a forward pass (evaluate the model on this training batch).
        # In PyTorch, calling `model` will in turn call the model's `forward` 
        # function and pass down the arguments. The `forward` function is 
        # documented here: 
        # https://huggingface.co/transformers/model_doc/bert.html#bertforsequenceclassification
        # The results are returned in a results object, documented here:
        # https://huggingface.co/transformers/main_classes/output.html#transformers.modeling_outputs.SequenceClassifierOutput
        # Specifically, we'll get the loss (because we provided labels) and the
        # "logits"--the model outputs prior to activation.
        result1 = model(b_input_ids_l1, 
                       token_type_ids=None, 
                       attention_mask=b_input_mask_l1, 
                       labels=b_labels_l1,
                       return_dict=True)
        
        result2 = model(b_input_ids_l2, 
                       token_type_ids=None, 
                       attention_mask=b_input_mask_l2, 
                       labels=b_labels_l2,
                       return_dict=True)
        
 

 
        loss1 = result1.loss
        loss2 = result2.loss
        

        logits1 = result1.logits
        logits2 = result2.logits

        eps = 1e-7
        # t = F.relu()
        # t = torch.log(t +eps)

        kthlogits1 = logits1[: , len(logits1[0])-1:len(logits1[0])].clone()
        kthlogits2 = logits2[: , len(logits2[0])-1:len(logits2[0])].clone()
      #  print("kthlogits before Relu   ",kthlogits1)
        ReLU = nn.ReLU()
        kthlogits1 = ReLU(kthlogits1)
      #  print("kthlogits after Relu   ",kthlogits1)
        kthloss1 = -1 * torch.log(kthlogits1 + eps)
       # print("kthloss  ",kthloss1)
        kthloss1 = torch.sum(kthloss1)
        kthloss1 = torch.sum(kthloss1)/len(b_labels_l1)
       # print("kthloss  ",kthloss1)
        #print("logits : ",logits)
        #print(len(logits))
        kthlogits2 = ReLU(kthlogits2)
        kthloss2 = -1 * torch.log(kthlogits2 + eps)
        kthloss2 = torch.sum(kthloss2)/len(b_labels_l2)
        loss = (loss2) + (0.2*kthloss2)+ (0.8*loss1) + (0.2*kthloss1) 

        # Accumulate the training loss over all of the batches so that we can
        # calculate the average loss at the end. `loss` is a Tensor containing a
        # single value; the `.item()` function just returns the Python value 
        # from the tensor.
        total_train_loss += loss.item()
        #print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        #print("GPU Usage before loss.backward()")
        #gpu_usage()

        # Perform a backward pass to calculate the gradients.
        loss.backward()

        # Clip the norm of the gradients to 1.0.
        # This is to help prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Update parameters and take a step using the computed gradient.
        # The optimizer dictates the "update rule"--how the parameters are
        # modified based on their gradients, the learning rate, etc.
        optimizer.step()

        # Update the learning rate.
        scheduler.step()
        #print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        #gpu_usage()
        
        del b_input_ids, b_input_mask, b_labels, b_input_ids_l1, b_input_mask_l1, b_labels_l1, b_input_ids_l2, b_input_mask_l2, b_labels_l2, loss, result1, result2
        gc.collect()
        torch.cuda.empty_cache()
        
        #print("GPU Usage after emptying the cache")
        #gpu_usage()
        #print("*****************************************************************************************")
        

    # Calculate the average loss over all of the batches.
    avg_train_loss = total_train_loss / len(train_dataloader)            
    
    # Measure how long this epoch took.
    training_time = format_time(time.time() - t0)

    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epcoh took: {:}".format(training_time))
        
    # ========================================
    #               Validation
    # ========================================
    # After the completion of each training epoch, measure our performance on
    # our validation set.

    print("")
    print("Running Validation...")

    t0 = time.time()

    # Put the model in evaluation mode--the dropout layers behave differently
    # during evaluation.
    model.eval()

    # Tracking variables 
    total_eval_accuracy = 0
    total_eval_loss = 0
    nb_eval_steps = 0

    # Evaluate data for one epoch
    for batch in validation_dataloader:
        
        # Unpack this training batch from our dataloader. 
        #
        # As we unpack the batch, we'll also copy each tensor to the GPU using 
        # the `to` method.
        #
        # `batch` contains three pytorch tensors:
        #   [0]: input ids 
        #   [1]: attention masks
        #   [2]: labels 
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        
        # Tell pytorch not to bother with constructing the compute graph during
        # the forward pass, since this is only needed for backprop (training).
        with torch.no_grad():        

            # Forward pass, calculate logit predictions.
            # token_type_ids is the same as the "segment ids", which 
            # differentiates sentence 1 and 2 in 2-sentence tasks.
            result = model(b_input_ids, 
                           token_type_ids=None, 
                           attention_mask=b_input_mask,
                           labels=b_labels,
                           return_dict=True)

        # Get the loss and "logits" output by the model. The "logits" are the 
        # output values prior to applying an activation function like the 
        # softmax.
        loss = result.loss
        logits = result.logits
       # print(logits)
        
        # Accumulate the validation loss.
        total_eval_loss += loss.item()

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
      #  print(label_ids) 
        # Calculate the accuracy for this batch of test sentences, and
        # accumulate it over all batches.
        total_eval_accuracy += flat_accuracy(logits, label_ids)
        

    # Report the final accuracy for this validation run.
    avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
    print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

    # Calculate the average loss over all of the batches.
    avg_val_loss = total_eval_loss / len(validation_dataloader)
    
    # Measure how long the validation run took.
    validation_time = format_time(time.time() - t0)
    
    print("  Validation Loss: {0:.2f}".format(avg_val_loss))
    print("  Validation took: {:}".format(validation_time))

    # Record all statistics from this epoch.
    training_stats.append(
        {
            'epoch': epoch_i + 1,
            'Training Loss': avg_train_loss,
            'Valid. Loss': avg_val_loss,
            'Valid. Accur.': avg_val_accuracy,
            'Training Time': training_time,
            'Validation Time': validation_time
        }
    )

    print("################################# Saving Model #####################################")

    output_dir = "/home/vkesana/model_save_open_set_cls6_opn"+str(epoch_i+1)+"_eph/"
    
    # Create output directory if needed
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(output_dir+'tokenizer/'):
        os.makedirs(output_dir+'tokenizer/')
        
    print("Saving model to %s" % output_dir)
    
    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir+'tokenizer/')



    print("################################# Testing ##########################################")
    try:
        test_function(tokenizer, model, device,epoch_i+1)
    except Exception as e:
        print(e)
    

print("")
print("Training complete!")

print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))






print("Running Validation...")

t0 = time.time()

# Put the model in evaluation mode--the dropout layers behave differently
# during evaluation.
model.eval()

# Tracking variables 
total_eval_accuracy = 0
total_eval_loss = 0
nb_eval_steps = 0

predictions , true_labels = [], []
count = 1
# Evaluate data for one epoch
for batch in validation_dataloader:
    
    # Unpack this training batch from our dataloader. 
    #
    # As we unpack the batch, we'll also copy each tensor to the GPU using 
    # the `to` method.
    #
    # `batch` contains three pytorch tensors:
    #   [0]: input ids 
    #   [1]: attention masks
    #   [2]: labels 
    b_input_ids = batch[0].to(device)
    b_input_mask = batch[1].to(device)
    b_labels = batch[2].to(device)

    
    # Tell pytorch not to bother with constructing the compute graph during
    # the forward pass, since this is only needed for backprop (training).
    with torch.no_grad():        

        # Forward pass, calculate logit predictions.
        # token_type_ids is the same as the "segment ids", which 
        # differentiates sentence 1 and 2 in 2-sentence tasks.
        result = model(b_input_ids, 
                        token_type_ids=None, 
                        attention_mask=b_input_mask,
                        labels=b_labels,
                        return_dict=True)

    # Get the loss and "logits" output by the model. The "logits" are the 
    # output values prior to applying an activation function like the 
    # softmax.
    loss = result.loss
    logits = result.logits
    # print(logits)
    
    # Accumulate the validation loss.
    total_eval_loss += loss.item()

    # Move logits and labels to CPU
    logits = logits.detach().cpu().numpy()
    logits[:,100] = logits[:,100] + 3
    label_ids = b_labels.to('cpu').numpy()
    pred_labels = np.argmax(logits, axis=1)
  #  print(label_ids) 
    # Calculate the accuracy for this batch of test sentences, and
    # accumulate it over all batches.

    total_eval_accuracy += flat_accuracy(logits, label_ids)

      # Store predictions and true labels
    predictions.extend(pred_labels.tolist())
    true_labels.extend(label_ids.tolist())
    # if(count == 1):
    #   break
    count = count +1

    

# Report the final accuracy for this validation run.
avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

# Calculate the average loss over all of the batches.
avg_val_loss = total_eval_loss / len(validation_dataloader)

# Measure how long the validation run took.
validation_time = format_time(time.time() - t0)

print("  Validation Loss: {0:.2f}".format(avg_val_loss))
print("  Validation took: {:}".format(validation_time))
training_stats.append(
        {
            'epoch': epoch_i + 1,
            'Training Loss': avg_train_loss,
            'Valid. Loss': avg_val_loss,
            'Valid. Accur.': avg_val_accuracy,
            'Training Time': training_time,
            'Validation Time': validation_time
        }
    )
print("")
print("Training complete!")
print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))



