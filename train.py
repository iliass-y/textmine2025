import pandas as pd
import numpy as np
import json
import torch
from torch import nn
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from sklearn.preprocessing import MultiLabelBinarizer
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, classification_report
from collections import Counter
import random
import os
import spacy
import pprint
import contextlib


os.environ["TOKENIZERS_PARALLELISM"] = "false"


class RelationExtractionDataset(Dataset):
    def __init__(self, texts, entity_pairs, labels, tokenizer, max_length, unique_relations=None):
        self.texts = texts
        self.entity_pairs = entity_pairs
        self.labels = labels  # Multi-label tensor
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.nlp = spacy.load('fr_core_news_lg', disable=["ner"])
        self.unique_relations = unique_relations

    def __len__(self):
        return len(self.entity_pairs)

    def __getitem__(self, idx):
        text = self.texts[idx]
        e1 = self.entity_pairs[idx][0]
        e2 = self.entity_pairs[idx][1]
        label = self.labels[idx]  # Multi-hot encoded label

        # Preprocessing
        #text = text.lower()

        marked_text = self.mark_entities_with_roles_and_all_mentions_rel_hints(
            text, e1, e2, unique_relations=self.unique_relations
        )

        #text = text.lower()


        encoding = self.tokenizer(
            marked_text,
            padding='max_length',
            truncation=False,
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.float)
        }



    def mark_entities_with_roles_and_all_mentions_rel_hints(self, text, e1, e2, role1="e1", role2="e2", 
                            self_reflective_role="se", unique_relations=None, max_mentions=None):
        """
        Marks entities with roles and includes possible relation hints at the end of the text.
        Allows limiting the number of mentions inserted by the 'max_mentions' parameter.
        """
        # Get all mentions of e1 and e2 sorted by their start position
        e1_mentions = sorted(e1['mentions'], key=lambda x: x['start'])
        e2_mentions = sorted(e2['mentions'], key=lambda x: x['start'])

        # If max_mentions is set, limit the number of mentions
        if max_mentions:
            e1_mentions = e1_mentions[:max_mentions]
            e2_mentions = e2_mentions[:max_mentions]

        # Prepare a list to track the insertion points and types
        insertions = []

        # If e1 and e2 refer to the same entity, use the self-reflective_role marker
        if e1['id'] == e2['id']:
            # Mark all mentions of this entity as self-reflective
            for mention in e1_mentions:
                insertions.append((mention['start'], mention['end'], self_reflective_role, e1['type']))
        else:
            # Different entities: prepare insertion points for e1 and e2 separately
            for mention in e1_mentions:
                insertions.append((mention['start'], mention['end'], role1, e1['type']))
            for mention in e2_mentions:
                insertions.append((mention['start'], mention['end'], role2, e2['type']))

        # Sort the insertions by start position in reverse to avoid messing up the indices
        insertions.sort(key=lambda x: x[0], reverse=True)

        # Modify the text by inserting the markers
        marked_text = text
        for start, end, role, entity_type in insertions:
            
            #lower entity types 
            #entity_type = entity_type.lower()

            # Use context-fitting markers
            marker_start = f"\n[{role}:{entity_type}]\n"
            #marker_start = f"<{role}>"
            #marker_end = f"</{role}>"  # Simplified marker end without type to fit context
            marker_end = f"\n[/{role}:{entity_type}]\n" #now that we use tokens it's okay to add type in the closing as well
            #marker_end = f"<{role}>"


            marked_text = marked_text[:start] + marker_start + marked_text[start:end] + marker_end + marked_text[end:]

        # Get possible relations to place hints
        #possible_relations = self.get_possible_relations(e1['type'], e2['type'], unique_relations)
        #possible_relations = self.get_possible_relations(e1['type'].lower(), e2['type'].lower(), unique_relations)
        possible_relations = self.get_possible_relations(e1['type'], e2['type'], unique_relations)

        #add prefix and suffix for relations (relation marking) :
        possible_relations = [f"[rel:{rel}]" for rel in possible_relations]


        
        # Now add the possible relation hints at the end of the text
        if possible_relations:
            # If e1 and e2 refer to the same entity, the only possible relations are gender male and gender female
            if e1['id'] == e2['id']:
                pass
                marked_text += f"\n[{role1}:{e1['type']}]\n[rel:GENDER_MALE]\n[rel:GENDER_FEMALE]\n[rel:no_relation]\n[/{role2}:{e2['type']}]\n."
                #marked_text += f"\n[{e1['type']}]\n[rel:GENDER_MALE]\n[rel:GENDER_FEMALE]\n[rel:no_relation]\n[/{e2['type']}]\n."
                #marked_text += f"\n[rel:GENDER_MALE]\n[rel:GENDER_FEMALE]"
            else:    
                relation_hints = '\n'.join(possible_relations)
                marked_text += f"\n[{role1}:{e1['type']}]\n{relation_hints}\n[rel:no_relation]\n[/{role2}:{e2['type']}]\n."
                #marked_text += f"\n[{e1['type']}]\n{relation_hints}\n[rel:no_relation]\n[/{e2['type']}]\n."
                #marked_text += f"\n{relation_hints}"
        
        return marked_text

    def get_possible_relations(self, e1_type, e2_type, possible_relations_list):
        relations = []
        for rel_tuple in possible_relations_list:
            entity1_type, relation, entity2_type = rel_tuple
            if entity1_type == e1_type and entity2_type == e2_type:
                relations.append(relation)
        return relations


def load_data(file_path):
    df = pd.read_csv(file_path)
    texts = df['text'].tolist()
    entities_list = df['entities'].apply(json.loads).tolist()

    def get_first_mention(entities):
        for ent in entities:
            ent['mentions'] = sorted(ent['mentions'], key=lambda x: x['start'])
        return entities

    entities_list = [get_first_mention(entities) for entities in entities_list]

    if 'relations' in df.columns:
        relations_list = df['relations'].apply(lambda x: json.loads(x) if pd.notna(x) else []).tolist()
    else:
        relations_list = [None] * len(df)

    return texts, entities_list, relations_list


def extract_unique_relations(texts, entities_list, relations_list):
    unique_relations = set()
    for idx, relations in enumerate(relations_list):
        if not relations:
            continue
        entities = entities_list[idx]
        for relation in relations:
            try:
                head_id, relation_type, tail_id = relation
                head_ent = next((ent for ent in entities if ent['id'] == head_id), None)
                tail_ent = next((ent for ent in entities if ent['id'] == tail_id), None)
                if head_ent is None or tail_ent is None:
                    continue
                head_type = head_ent['type']
                tail_type = tail_ent['type']
                relation_tuple = (head_type, relation_type, tail_type)
                unique_relations.add(relation_tuple)
            except Exception:
                continue
    return unique_relations


def save_relations(unique_relations, output_path, format='json'):
    if format == 'json':
        relations_list = [list(rel) for rel in unique_relations]
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(relations_list, f, ensure_ascii=False, indent=4)
    elif format == 'csv':
        df = pd.DataFrame(list(unique_relations), columns=['head_ent_type', 'relation_type', 'tail_ent_type'])
        df.to_csv(output_path, index=False, encoding='utf-8')
    else:
        raise ValueError("Unsupported format. Choose 'json' or 'csv'.")


def load_relations_from_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        relations_list = json.load(f)
    relations = [tuple(rel) for rel in relations_list]
    return relations


def generate_relation_tokens(unique_relations, self_reflective_types):
    tokens_to_add = set()
    for relation in unique_relations:
        e1_type, rel_name, e2_type = relation
        tokens_to_add.add(f"[e1:{e1_type}]")
        tokens_to_add.add(f"[e2:{e2_type}]")
        tokens_to_add.add(f"[/e1:{e1_type}]")
        tokens_to_add.add(f"[/e2:{e2_type}]")
        tokens_to_add.add(f"[rel:{rel_name}]")
        if e1_type == e2_type and e1_type in self_reflective_types:
            tokens_to_add.add(f"[se:{e1_type}]")
            tokens_to_add.add(f"[/se:{e2_type}]")
    tokens_to_add.add(f"[rel:no_relation]")
    return list(tokens_to_add)


def prepare_dataset_with_local_negative_sampling(texts, entities_list, relations_list, mlb, possible_relations,
                                                 negative_ratio=5, generate_negatives=True):
    all_texts = []
    all_entity_pairs = []
    all_labels = []
    label_counter = Counter()

    for text, entities, relations in zip(texts, entities_list, relations_list):
        entity_dict = {}
        for ent in entities:
            mention = ent['mentions'][0]
            entity_dict[ent['id']] = {
                'id': ent['id'],
                'value': mention['value'],
                'start': mention['start'],
                'end': mention['end'],
                'type': ent['type'],
                'mentions': ent['mentions']
            }
        entity_ids = list(entity_dict.keys())
        positive_pairs = {}
        # Collect positive relations
        for rel in relations:
            head_id, rel_type, tail_id = rel
            if head_id in entity_dict and tail_id in entity_dict:
                head_ent = entity_dict[head_id]
                tail_ent = entity_dict[tail_id]
                pair_key = (head_ent['id'], tail_ent['id'])
                if pair_key not in positive_pairs:
                    positive_pairs[pair_key] = set()
                positive_pairs[pair_key].add(rel_type)
                label_counter.update([rel_type])

        # Prepare data for positive pairs
        for pair_key, rel_types in positive_pairs.items():
            head_id, tail_id = pair_key
            head_ent = entity_dict[head_id]
            tail_ent = entity_dict[tail_id]
            all_texts.append(text)
            all_entity_pairs.append((head_ent, tail_ent))
            all_labels.append(list(rel_types))

        # Generate negatives
        if generate_negatives:
            negative_samples = []
            for i in range(len(entity_ids)):
                for j in range(len(entity_ids)):
                    if i == j or (entity_ids[i], entity_ids[j]) in positive_pairs:
                        continue
                    head_id = entity_ids[i]
                    tail_id = entity_ids[j]
                    head_ent = entity_dict[head_id]
                    tail_ent = entity_dict[tail_id]
                    # Check if relation is possible
                    relation_possible = False
                    for subj_type, rel_name, obj_type in possible_relations:
                        if head_ent['type'] == subj_type and tail_ent['type'] == obj_type:
                            relation_possible = True
                            break
                    if relation_possible:
                        negative_samples.append((text, (head_ent, tail_ent), ['no_relation']))

            num_positive_pairs = len(positive_pairs)
            num_negatives_to_generate = negative_ratio * num_positive_pairs
            num_negatives = min(num_negatives_to_generate, len(negative_samples))

            hard_negatives = random.sample(negative_samples, num_negatives)
            for neg_sample in hard_negatives:
                all_texts.append(neg_sample[0])
                all_entity_pairs.append(neg_sample[1])
                all_labels.append(neg_sample[2])
                label_counter.update(neg_sample[2])

    # Transform labels using mlb
    all_labels = mlb.transform(all_labels)

    return all_texts, all_entity_pairs, all_labels, label_counter


def stratified_split(X, y, entity_pairs, test_size=0.2, random_state=42):
    msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    for train_index, test_index in msss.split(X, y):
        X_train, X_val = [X[i] for i in train_index], [X[i] for i in test_index]
        y_train, y_val = y[train_index], y[test_index]
        entity_pairs_train = [entity_pairs[i] for i in train_index]
        entity_pairs_val = [entity_pairs[i] for i in test_index]
    return X_train, X_val, y_train, y_val, entity_pairs_train, entity_pairs_val


def get_label_distribution(y, mlb):
    label_counts = y.sum(axis=0)
    label_distribution = label_counts / y.shape[0]
    return dict(zip(mlb.classes_, label_distribution))


def compute_metrics_multi_per_class_thresholds(p, thresholds, mlb):
    probs = torch.sigmoid(torch.tensor(p.predictions)).numpy()
    preds = (probs >= thresholds).astype(int)
    labels = p.label_ids

    macro_f1 = f1_score(labels, preds, average='macro', zero_division=0)
    precision = precision_score(labels, preds, average='macro', zero_division=0)
    recall = recall_score(labels, preds, average='macro', zero_division=0)
    accuracy = accuracy_score(labels, preds)

    # Per-class F1 scores
    classification_rep = classification_report(
        labels, preds, target_names=mlb.classes_, output_dict=True, zero_division=0
    )
    print("Per-class F1 Scores:")
    pprint.pprint(classification_rep)

    return {
        'macro_f1': macro_f1,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall
    }


#------------------------------------------


class AsymmetricLossOptimized(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8,
                 disable_torch_grad_focal_loss=False, class_weights=None):
        super(AsymmetricLossOptimized, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad = disable_torch_grad_focal_loss
        self.eps = eps
        self.class_weights = class_weights  # Add class weights

    def forward(self, x, y):
        targets = y
        anti_targets = 1 - y

        # Calculating Probabilities
        xs_pos = torch.sigmoid(x)
        xs_neg = 1.0 - xs_pos

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        loss = targets * torch.log(xs_pos.clamp(min=self.eps))
        loss += anti_targets * torch.log(xs_neg.clamp(min=self.eps))

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            with torch.no_grad() if self.disable_torch_grad else contextlib.nullcontext():
                pt = xs_pos * targets + xs_neg * anti_targets  # pt is the model's estimated probability for the true class
                one_sided_gamma = self.gamma_pos * targets + self.gamma_neg * anti_targets
                focal_weight = (1 - pt) ** one_sided_gamma

            loss *= focal_weight

        # Apply class weights
        if self.class_weights is not None:
            # Reshape class weights to match the shape of loss
            cw = self.class_weights.unsqueeze(0)  # Shape: [1, num_classes]
            loss = loss * cw

        return -loss.mean()
#------------------------------------------------------------------------------------


def main():
    # Load data
    train_texts, train_entities_list, train_relations_list = load_data('train.csv')

    all_relations = set()
    for relations in train_relations_list:
        if relations is not None:
            for rel in relations:
                all_relations.add(rel[1])
    all_relations.add('no_relation')

    unique_relations = extract_unique_relations(train_texts, train_entities_list, train_relations_list)
    unique_relations = sorted(unique_relations)

    # Entity types that get SE:type marker for self-reflective relations
    self_reflective_types = {"CIVILIAN", "TERRORIST_OR_CRIMINAL", "MILITARY"}

    # Save unique relations
    output_dir = 'output_relations'
    os.makedirs(output_dir, exist_ok=True)
    json_output_path = os.path.join(output_dir, 'unique_relations.json')
    save_relations(unique_relations, json_output_path, format='json')

    # MultiLabelBinarizer for multi-label encoding
    mlb = MultiLabelBinarizer()
    mlb.fit([list(all_relations)])

    # Save the label encoder for later use
    with open('mlb.json', 'w') as f:
        json.dump({'classes': mlb.classes_.tolist()}, f)

    # Prepare training data
    train_texts_prepared, train_entity_pairs, train_labels, label_counter = prepare_dataset_with_local_negative_sampling(
        train_texts, train_entities_list, train_relations_list, mlb, unique_relations,
        generate_negatives=True,
        negative_ratio=3
    )

    print("class count :")
    print(label_counter)

    # Convert labels to NumPy array
    train_labels = np.array(train_labels)

    # Perform stratified split
    X_train, X_val, y_train, y_val, entity_pairs_train, entity_pairs_val = stratified_split(
        train_texts_prepared, train_labels, train_entity_pairs, test_size=0.2, random_state=42
    )

    # Calculate label distributions
    train_label_distribution = get_label_distribution(y_train, mlb)
    val_label_distribution = get_label_distribution(y_val, mlb)

    print("Training label distribution:")
    print(train_label_distribution)
    print("\nValidation label distribution:")
    print(val_label_distribution)

    # Load tokenizer and model
    base_model_name = "cmarkea/distilcamembert-base"
 
    #base_model_name = "flaubert/flaubert_base_cased"

    #base_model_name = "FacebookAI/xlm-roberta-base"

    #base_model_name = "camembert/camembert-base-ccnet"
    


    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    # Generate tokens based on the unique relations
    tokens_to_extend = generate_relation_tokens(unique_relations, self_reflective_types)
    print(f"Added tokens: {tokens_to_extend}")

    special_tokens_dict = {'additional_special_tokens': tokens_to_extend}
    
    tokenizer.add_special_tokens(special_tokens_dict)

    model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        num_labels=len(mlb.classes_),  # Multi-label classification
    )

    model.resize_token_embeddings(len(tokenizer))

    # Adjust dropout rates
    model.config.hidden_dropout_prob = 0.3,
    model.config.attention_probs_dropout_prob = 0.3
    model.config.classifier_dropout = 0.3

    train_dataset = RelationExtractionDataset(
        X_train, entity_pairs_train, y_train, tokenizer, 
        max_length=512,
        # max_length=768, # because gte can handle up to 2048 tokens
          unique_relations=unique_relations
    )
    val_dataset = RelationExtractionDataset(
        X_val, entity_pairs_val, y_val, tokenizer,
        max_length=512, 
        #max_length=768,# because gte can handle up to 2048 tokens
         
          unique_relations=unique_relations
    )

    # Compute class weights using Effective Number of Samples
    def compute_class_weights(label_counts, beta=0.9999):
        effective_num = 1.0 - np.power(beta, label_counts)
        weights = (1.0 - beta) / np.array(effective_num)
        weights = weights / np.sum(weights) * len(label_counts)
        return torch.tensor(weights, dtype=torch.float32)

    label_counts = y_train.sum(axis=0)
    class_weights_tensor = compute_class_weights(label_counts)

    # Adjust weights for 'no_relation' class
    no_relation_idx = np.where(mlb.classes_ == 'no_relation')[0][0]
    class_weights_tensor[no_relation_idx] = class_weights_tensor.min()

    # Implement per-class thresholds
    def find_optimal_thresholds(y_true, y_probs, mlb):
        thresholds = np.linspace(0.1, 0.9, 81)
        best_thresholds = np.zeros(y_probs.shape[1])
        for i in range(y_probs.shape[1]):
            f1_scores = []
            for thresh in thresholds:
                preds = (y_probs[:, i] >= thresh).astype(int)
                f1 = f1_score(y_true[:, i], preds, zero_division=0)
                f1_scores.append(f1)
            best_thresh = thresholds[np.argmax(f1_scores)]
            best_thresholds[i] = best_thresh
        return best_thresholds

    # Define custom compute_metrics function
    class MetricsCallback:
        def __init__(self, mlb):
            self.mlb = mlb
            self.best_thresholds = None

        def compute_metrics(self, p):
            probs = torch.sigmoid(torch.tensor(p.predictions)).numpy()
            labels = p.label_ids
            #need to calculate treshold every single time
            #if self.best_thresholds is None:
            # Find optimal thresholds on validation set
            self.best_thresholds = find_optimal_thresholds(labels, probs, self.mlb)
            print(self.best_thresholds)
            return compute_metrics_multi_per_class_thresholds(p, self.best_thresholds, self.mlb)

    metrics_callback = MetricsCallback(mlb)

    training_args = TrainingArguments(
        output_dir=f'./results/{base_model_name}',
        num_train_epochs=10,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=3000,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        save_total_limit=3,
        metric_for_best_model="eval_macro_f1",
        fp16=True,
        #bf16 = True,
        dataloader_num_workers=48,

        #gradient_accumulation_steps=2,
        #optim ="schedule_free_adamw",
        #optim ="adamw_bnb_8bit"
    )

    # Define custom loss function (Focal Loss)
    class FocalLoss(nn.Module):
        def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
            super(FocalLoss, self).__init__()
            self.alpha = alpha
            self.gamma = gamma
            self.reduction = reduction

        def forward(self, inputs, targets):
            BCE_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
            pt = torch.exp(-BCE_loss)
            F_loss = (self.alpha * (1 - pt) ** self.gamma * BCE_loss).mean()
            return F_loss

    # Create trainer class with custom loss
    class CustomTrainer(Trainer):
        def __init__(self, *args, **kwargs):
            self.class_weights = kwargs.pop('class_weights')
            super().__init__(*args, **kwargs)
            self.loss_fn = FocalLoss(alpha=self.class_weights.to(self.args.device), gamma=2.0)

        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            loss = self.loss_fn(logits, labels)
            return (loss, outputs) if return_outputs else loss

    #---------------------------------------------------------------------------------------------------------------


        #different asymmetric loss trainer--------------------------------------------------------------------------------------------------------------------------
    
    class AsymmetricLossOptimizedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            
            # Calculate loss using ASL
            loss = asl_loss_fn(logits, labels)
            
            return (loss, outputs) if return_outputs else loss



    #asl_loss_fn = AsymmetricLossOptimized( gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=False
    #,class_weights=class_weights_tensor
    #,class_weights=None
    #)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    asl_loss_fn = AsymmetricLossOptimized( gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=False
    ,class_weights=class_weights_tensor.to(device)
    #,class_weights=None
    )


    
    trainer = AsymmetricLossOptimizedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        #compute_metrics=compute_metrics_multi,
        compute_metrics=metrics_callback.compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        #class_weights=class_weights_tensor
    )

    #assym loss with label smoothing---------------------------------------------------------------------------------------------------------------------------

    #------------------------------------------------------------------
    """
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=metrics_callback.compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        class_weights=class_weights_tensor
    )
    """
    # Freeze lower layers (optional)
    #for name, param in model.named_parameters():
    #    if 'layer' in name:
    #        layer_num = int(name.split('.')[2])
    #        if layer_num < 6:  # Freeze layers 0-5
    #            param.requires_grad = False

    trainer.train()

    # Save the best thresholds after training
    np.save('best_thresholds.npy', metrics_callback.best_thresholds)
    trainer.save_model(f'{base_model_name}-finetuned')

    # Save tokenizer
    tokenizer.save_pretrained(f'{base_model_name}-finetuned')

if __name__ == '__main__':
    main()
