import torch
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

def test_function(tokenizer, model, device,epoch):


    df = pd.read_csv("./Data/ood_test_data_small.csv")
    df = df[['real_label','review']]
    arr_real_labels = [i for i in range(100,10000)]
    arr_real_labels
    df['real_label'] = df['real_label'].replace(arr_real_labels,100)
    df=df.loc[df.real_label.values<=100]
    df.groupby(by='real_label').count()
    labels_list_0 = df['real_label'].tolist()


    labels = df.real_label.values
    sentence1 = df.review.values
    #sentence1=sentences
    #labels = dataset_filtered.real_label.values

    # Tokenize all of the sentences and map the tokens to thier word IDs.
    input_ids = []
    attention_masks = []


    for sent1 in zip(sentence1):

        encoded_dict = tokenizer.encode_plus(
                            sent1[0],# Sentence to encode.
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



    # Set the batch size.  
    batch_size = 16

    # Create the DataLoader.
    prediction_data = TensorDataset(input_ids, attention_masks, labels)
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)

    # Prediction on test set

    print('Predicting labels for {:,} test sentences...'.format(len(input_ids)))

    # Put model in evaluation mode
    model.eval()

    # Tracking variables 
    predictions , true_labels = [], []
    logits_arr=[]
    max_logits_arr=[]
    logits_total_array = []

# Predict 
    for batch in prediction_dataloader:
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)
        
        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch
        
        # Telling the model not to compute or store gradients, saving memory and 
        # speeding up prediction
        with torch.no_grad():
            # Forward pass, calculate logit predictions.
            result = model(b_input_ids, 
                            token_type_ids=None, 
                            attention_mask=b_input_mask,
                            return_dict=True)

        logits = result.logits

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        logits_total_array.extend(logits.tolist())
        #logits[:,100] = logits[:,100] + 2.5
        pred_labels = np.argmax(logits, axis=1)
        label_ids = b_labels.to('cpu').numpy()
        max_logits = np.max(logits[:,:100], axis=1)
        # Store predictions and true labels
        predictions.extend(pred_labels.tolist())
        true_labels.extend(label_ids.tolist())
        logits_arr.extend(logits[:,100].tolist())
        max_logits_arr.extend(max_logits.tolist())

    print('    DONE.')

    d1_predictions = predictions
    d1_true_labels = true_labels
    d1_logits_arr = logits_arr
    d1_max_logits_arr = max_logits_arr

    print(d1_predictions.count(100))

    tic = 20000

    d1_closed_logits=[]
    d1_open_logits=[]
    d1_max_closed_logits=[]
    d1_max_open_logits=[]
    d1_predictions_open=[]
    d1_predictions_closed=[]
    for i in range(tic):
        if(true_labels[i]==100):
            d1_open_logits.append(d1_logits_arr[i])
            d1_max_open_logits.append(d1_max_logits_arr[i])
            d1_predictions_open.append(d1_predictions[i])
        else:
            d1_closed_logits.append(d1_logits_arr[i])
            d1_max_closed_logits.append(d1_max_logits_arr[i])
            d1_predictions_closed.append(d1_predictions[i])

#print(d1_predictions_open.count(100))
#print(d1_predictions_closed.count(100))

    diff_closed1=[]
    for i in range(int(tic/2)):
        diff_closed1.append(5.08 + d1_closed_logits[i] - d1_max_closed_logits[i])
    diff_open1=[]
    for i in range(int(tic/2)):
          diff_open1.append(5.08 + d1_open_logits[i] - d1_max_open_logits[i])

#print(sum(diff_closed1))
#print(sorted(diff_closed1)[int(tic/4)])

#print(sum(diff_open1))
#print(sorted(diff_open1)[int(tic/4)])

    pos_count = 0 
    neg_count = 0
    for num in diff_closed1:
        # checking condition
        if num >= 0:
            pos_count += 1
        else:
            neg_count += 1

#print(pos_count)
#print(neg_count)

    pos_count = 0 
    neg_count = 0
    for num in diff_open1:
        # checking condition
        if num >= 0:
            pos_count += 1
        else:
            neg_count += 1

#print(pos_count)
#print(neg_count)



    df2 = pd.read_csv("./Data/ood_test_data_small.csv")


    combined_df = df2
    number_of_known_labels = 100

    def compute_precision_recall_for_known_classes(given_label):
        TP = combined_df[(combined_df["real_label"] == given_label) & (combined_df["N_plus_one_prediction"] == given_label)].shape[0]
        FP = combined_df[(combined_df["real_label"] != given_label) & (combined_df["N_plus_one_prediction"] == given_label)].shape[0]
        FN = combined_df[(combined_df["real_label"] == given_label) & (combined_df["N_plus_one_prediction"] != given_label)].shape[0]
        if(TP == 0):
            P = 0
            R = 0
        else:
            P = round(100*TP/(TP + FP), 2)
            R = round(100*TP/(TP + FN),2)
        return P, R

    def compute_precision_recall_for_OOD():
        TP = combined_df[(combined_df["real_label"] >= number_of_known_labels) & (combined_df["N_plus_one_prediction"] == "OOD")].shape[0]
        FP = combined_df[(combined_df["real_label"] < number_of_known_labels) & (combined_df["N_plus_one_prediction"] == "OOD")].shape[0]
        FN = combined_df[(combined_df["real_label"] >= number_of_known_labels) & (combined_df["N_plus_one_prediction"] != "OOD")].shape[0]
        if(TP == 0):
            P = 0
            R = 0
        else:
            P = 100*TP/(TP + FP)
            R = 100*TP/(TP + FN)
        return P, R

    def compute_performance_metrics():
        from statistics import mean
        
        precisions = []
        recalls = []
        for known_label in range(number_of_known_labels):
            p, r = compute_precision_recall_for_known_classes(known_label)
            precisions.append(p)
            recalls.append(r)
        
        precision_known, recall_known = round(mean(precisions),2), round(mean(recalls),2)
        f1_known = round(2*precision_known*recall_known/(precision_known + recall_known),2)
        
        precision_ood, recall_ood = compute_precision_recall_for_OOD()
        f1_ood = round(2*precision_ood*recall_ood/(precision_ood + recall_ood),2)
        
        precisions.append(precision_ood)
        recalls.append(recall_ood)
        
        precision, recall = round(mean(precisions),2), round(mean(recalls),2)
        f1 = round(2*precision*recall/(precision + recall),2)
        
        return {
            "known": {
                "precision": precision_known,
                "recall": recall_known,
                "f1": f1_known,
            },
            "ood": {
                "precision": round(precision_ood,2),
                "recall": round(recall_ood,2),
                "f1": f1_ood,
            },
            "overall": {
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
            
        }

    def print_performance_metrics():
        performance_dict = compute_performance_metrics()
        print(performance_dict["known"]["precision"],performance_dict["known"]["recall"],performance_dict["known"]["f1"], end = " ")
        print(performance_dict["ood"]["precision"],performance_dict["ood"]["recall"],performance_dict["ood"]["f1"], end = " ")
        print(performance_dict["overall"]["precision"],performance_dict["overall"]["recall"],performance_dict["overall"]["f1"], end = " ")
        return performance_dict["overall"]["f1"]

    #print_performance_metrics()

    list_of_f1=[]
    c = 0
    for i in range(20):
        logits_total_array_o =logits_total_array
        logits_total_array_o = np.array(logits_total_array_o)
        logits_total_array_o[:,100] = logits_total_array_o[:,100] + (c)
        c = c + 0.5
        predictions_ = np.argmax(logits_total_array_o , axis=1)
        df2["N_plus_one_prediction"] = predictions_
        df2['N_plus_one_prediction'] = df2['N_plus_one_prediction'].replace([100], 'OOD')
        print("\n",c-0.5,"    ",end="")
        list_of_f1.append(print_performance_metrics())
        # print_performance_metrics()

    max(list_of_f1)


    logits_total_array_o = logits_total_array_o.tolist()

    df2['logits'] = logits_total_array_o

    #df2.to_csv("./logits_updated_2"+str(epoch)+".csv")