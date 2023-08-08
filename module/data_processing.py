import torch

def get_train_valid_dataset(training_args, tokenizer, model_config):
    # Load dataset
    from datasets import load_dataset
    dataset = load_dataset("voidful/NMSQA-CODE")
    train_dataset = dataset['train'].filter(lambda x: x['hubert_100_question_unit'] != '')
    valid_dataset = dataset['dev'].filter(lambda x: x['hubert_100_question_unit'] != '')

    def process_data_to_model_inputs(item):
        input_question = convert_vtok(item['hubert_100_question_unit'])
        input_context = convert_vtok(item['hubert_100_context_unit'])
        label_sent = convert_vtok(item['hubert_100_answer_unit'])

        input_sent_tokens = tokenizer.encode(input_question, input_context, return_tensors='pt',
                                                              add_special_tokens=False).to('cuda')
        label_sent_tokens = tokenizer.encode(label_sent, return_tensors='pt',
                                                    add_special_tokens=False).to('cuda')

        concatenated = torch.cat([input_sent_tokens, label_sent_tokens,
                                    torch.tensor([[tokenizer.eos_token_id]]).to('cuda')], dim=-1)
        labels = torch.cat([torch.full_like(input_sent_tokens, -100).to('cuda'), label_sent_tokens,
                            torch.tensor([[tokenizer.eos_token_id]]).to('cuda')], dim=-1)
        attention_mask = torch.ones_like(concatenated[:, :-1]).to('cuda')
        return {
            "input_ids": torch.flatten(concatenated[:, :-1]),
            "attention_mask": torch.flatten(attention_mask),
            "labels": torch.flatten(labels[:, 1:])
        }


    # Apply the processing function to the datasets
    train_dataset = train_dataset.map(
        process_data_to_model_inputs,
        cache_file_name="nmsqa-train",

    )
    valid_dataset = valid_dataset.map(
        process_data_to_model_inputs,
        cache_file_name="nmsqa-dev"
    )

    columns = ["input_ids", "labels", "attention_mask"]
    train_dataset.set_format(type="torch", columns=columns)
    valid_dataset.set_format(type="torch", columns=columns)
    print("train_dataset", train_dataset[0])
    print("valid_dataset", valid_dataset[0])

    return train_dataset, valid_dataset
import json
def convert_vtok(unit_code):
    try:
        code = json.loads(unit_code)[0]['merged_code']
    except:
        return
    v_tok = [f"v_tok_{unit}" for unit in code]
    unit_code = ' '.join(v_tok) # blank is not needed
    return unit_code