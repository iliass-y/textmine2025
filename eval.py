import pandas as pd
import numpy as np
import json
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm
import spacy
import os
import re

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class RelationExtractionDataset(Dataset):
    def __init__(self, texts, entity_pairs, labels, tokenizer, max_length, unique_relations=None):
        self.texts = texts
        self.entity_pairs = entity_pairs
        self.labels = labels  # Dummy labels, not used during prediction
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

        # Preprocessing Steps (match training code)
        #text = text.lower()
        marked_text = self.mark_entities_with_roles_and_all_mentions_rel_hints(
            text, e1, e2, unique_relations=self.unique_relations
        )

        # Tokenize the text
        encoding = self.tokenizer(
            marked_text,
            padding='max_length',
            truncation=False,
            max_length=self.max_length,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
        }

    """
    def mark_entities_with_roles_and_all_mentions_rel_hints(self, text, e1, e2, role1="e1", role2="e2",
                                                            self_reflective_role="se", unique_relations=None):
        e1_mentions = sorted(e1['mentions'], key=lambda x: x['start'])
        e2_mentions = sorted(e2['mentions'], key=lambda x: x['start'])

        insertions = []

        if e1['id'] == e2['id']:
            for mention in e1_mentions:
                insertions.append((mention['start'], mention['end'], self_reflective_role, e1['type']))
        else:
            for mention in e1_mentions:
                insertions.append((mention['start'], mention['end'], role1, e1['type']))
            for mention in e2_mentions:
                insertions.append((mention['start'], mention['end'], role2, e2['type']))

        insertions.sort(key=lambda x: x[0], reverse=True)

        marked_text = text
        for start, end, role, entity_type in insertions:
            marker_start = f"[{role}:{entity_type}]"
            marker_end = f"[/{role}:{entity_type}]"
            marked_text = marked_text[:start] + marker_start + marked_text[start:end] + marker_end + marked_text[end:]

        possible_relations = self.get_possible_relations(e1['type'], e2['type'], unique_relations)
        possible_relations = [f"[rel:{rel}]" for rel in possible_relations]

        if possible_relations:
            relation_hints = ' '.join(possible_relations)
            marked_text += f" {relation_hints}"

        return marked_text
    """

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
    ids = df['id'].tolist() if 'id' in df.columns else list(range(len(df)))
    texts = df['text'].tolist()
    entities_list = df['entities'].apply(json.loads).tolist()

    def get_first_mention(entities):
        for ent in entities:
            ent['mentions'] = sorted(ent['mentions'], key=lambda x: x['start'])
        return entities

    entities_list = [get_first_mention(entities) for entities in entities_list]
    return ids, texts, entities_list


def load_relations_from_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        relations_list = json.load(f)
    relations = [tuple(rel) for rel in relations_list]
    return relations


def get_class_index(mlb, class_name):
    if class_name in mlb.classes_:
        return np.where(mlb.classes_ == class_name)[0][0]
    else:
        return None


def prepare_dataset(texts, entities_list, possible_relations):
    all_texts = []
    all_entity_pairs = []
    text_indices = []
    entity_id_pairs = []

    for idx, (text, entities) in enumerate(zip(texts, entities_list)):
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

        # Generate all possible pairs
        for i in range(len(entity_ids)):
            for j in range(len(entity_ids)):
                head_id = entity_ids[i]
                tail_id = entity_ids[j]
                head_ent = entity_dict[head_id]
                tail_ent = entity_dict[tail_id]

                # Check if the relation is possible
                possible = False
                if head_id == tail_id:
                    if head_ent['type'] in ["CIVILIAN", "TERRORIST_OR_CRIMINAL", "MILITARY"]:
                        possible = True
                else:
                    for subj_type, _, obj_type in possible_relations:
                        if head_ent['type'] == subj_type and tail_ent['type'] == obj_type:
                            possible = True
                            break

                if possible:
                    all_texts.append(text)
                    all_entity_pairs.append((head_ent, tail_ent))
                    text_indices.append(idx)
                    entity_id_pairs.append((head_id, tail_id))

    return all_texts, all_entity_pairs, text_indices, entity_id_pairs


def main():
    # Load the trained model and tokenizer

    #replace with trained model path
    base_model = "best model ever"
    

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSequenceClassification.from_pretrained(base_model)
    model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    # Load label encoder classes
    with open('mlb.json', 'r') as f:
        label_encoder_data = json.load(f)
    mlb = MultiLabelBinarizer()
    mlb.classes_ = np.array(label_encoder_data['classes'])

    # Load evaluation data
    eval_ids, eval_texts, eval_entities_list = load_data('test_01-07-2024.csv')

    # Load unique relations
    json_path = 'output_relations/unique_relations.json'
    unique_relations = load_relations_from_json(json_path)

    # Prepare evaluation data
    eval_texts_prepared, eval_entity_pairs, text_indices, entity_id_pairs = prepare_dataset(
        eval_texts, eval_entities_list, unique_relations
    )

    # Create evaluation dataset
    max_length = 512  # Match with training
    eval_dataset = RelationExtractionDataset(
        eval_texts_prepared, eval_entity_pairs, [0] * len(eval_entity_pairs), tokenizer,
        max_length=max_length, unique_relations=unique_relations
    )

    # Load best thresholds computed during training
    #best_thresholds = np.load('./dynamic_thresh_classes_foc_exp/camembert_base/best_thresholds.npy')
    
    #load the threshold values obtained during training
    
    best_thresholds = np.load('optimal_tresholds.npy')

    # Make predictions
    model.eval()
    predictions = []
    probs_list = []
    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=64, num_workers=48)

    for batch in tqdm(eval_loader, desc="Predicting"):
        inputs = {
            'input_ids': batch['input_ids'].to(model.device),
            'attention_mask': batch['attention_mask'].to(model.device),
        }
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(logits).cpu().numpy()
            probs_list.append(probs)

    # Stack probabilities into a NumPy array
    probs = np.vstack(probs_list)

    # Apply per-class thresholds
    predictions = (probs >= best_thresholds).astype(int)

    # Convert adjusted predictions back to labels using mlb
    pred_relations_list = mlb.inverse_transform(predictions)

    # Create a lookup for possible relations
    possible_relation_lookup = {
        (subj_type, rel_name, obj_type) for subj_type, rel_name, obj_type in unique_relations
    }

    # Prepare submission
    submission_dict = {id_: [] for id_ in eval_ids}

    for idx, pred_relations in enumerate(pred_relations_list):
        text_idx = text_indices[idx]
        head_ent, tail_ent = eval_entity_pairs[idx]
        head_id, tail_id = entity_id_pairs[idx]
        eval_id = eval_ids[text_idx]

        for pred_relation in pred_relations:
            if pred_relation != 'no_relation':
                if pred_relation in ['GENDER_MALE', 'GENDER_FEMALE']:
                    if (head_id == tail_id
                        and head_ent['type'] in ["CIVILIAN", "TERRORIST_OR_CRIMINAL", "MILITARY"]
                        and tail_ent['type'] in ["CIVILIAN", "TERRORIST_OR_CRIMINAL", "MILITARY"]):
                        submission_dict[eval_id].append([head_id, pred_relation, tail_id])
                elif head_id != tail_id:
                    if (head_ent['type'], pred_relation, tail_ent['type']) in possible_relation_lookup:
                        submission_dict[eval_id].append([head_id, pred_relation, tail_id])

    # Save submission file
    submission_df = pd.DataFrame({
        'id': eval_ids,
        'relations': [json.dumps(submission_dict[id_]) for id_ in eval_ids]
    })
    submission_df.to_csv('submission.csv', index=False)


if __name__ == '__main__':
    main()
