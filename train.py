import json
import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup

class CODI_Dataset(Dataset) :

    def __init__(self, data_path, tokenizer, split = 'train', icot_length = 6, max_length = 256) :

        self.data_path = data_path
        self.tokenizer = tokenizer
        self.split = split
        self.icot_length = icot_length
        self.max_length = max_length

        if self.split == 'train' :
            with open(self.data_path, 'r', encoding = 'utf-8') as f :
                self.data = f.readlines()
        if self.split == 'test' :
            self.data = pd.read_parquet(self.data_path).to_dict(orient = 'records')
    
    def __len__(self) :

        return len(self.data)

    def __getitem__(self, idx) :

        if self.split == 'train' :

            question, answer = self.data[idx].strip().split('||')
            cot, answer = answer.split(' #### ')
            cot = ' '.join(cot.split(' ')[:-1])

            bos_input_ids = self.tokenizer('<|endoftext|>', return_tensors = 'pt', add_special_tokens = False)['input_ids']
            question_input_ids = self.tokenizer(question, return_tensors = 'pt', add_special_tokens = False)['input_ids']
            cot_input_ids = self.tokenizer(cot, return_tensors = 'pt', add_special_tokens = False)['input_ids']
            icot_input_ids = self.tokenizer('<bot>' + '<|endoftext|>' * self.icot_length + '<eot>', return_tensors = 'pt', add_special_tokens = False)['input_ids']
            symbol_input_ids = self.tokenizer('The Answer is:', return_tensors = 'pt', add_special_tokens = False)['input_ids']
            answer_input_ids = self.tokenizer(answer, return_tensors = 'pt', add_special_tokens = False)['input_ids']

            if len(cot) != 0 :
                teacher_input_ids = torch.cat([bos_input_ids, question_input_ids, cot_input_ids, symbol_input_ids, answer_input_ids], dim = 1)
            else :
                teacher_input_ids = torch.cat([bos_input_ids, question_input_ids, symbol_input_ids, answer_input_ids], dim = 1)
            student_input_ids = torch.cat([bos_input_ids, question_input_ids, icot_input_ids, symbol_input_ids, answer_input_ids], dim = 1)

            teacher_pad_input_ids = torch.cat([bos_input_ids for i in range(self.max_length - len(teacher_input_ids[0]))], dim = 1)
            student_pad_input_ids = torch.cat([bos_input_ids for i in range(self.max_length - len(student_input_ids[0]))], dim = 1)
            teacher_input_ids = torch.cat([teacher_input_ids, teacher_pad_input_ids], dim = 1)
            student_input_ids = torch.cat([student_input_ids, student_pad_input_ids], dim = 1)

            teacher_attention_mask = torch.ones_like(teacher_input_ids)
            teacher_attention_mask[:, -len(teacher_pad_input_ids[0]):] = 0
            student_attention_mask = torch.ones_like(student_input_ids)
            student_attention_mask[:, -len(student_pad_input_ids[0]):] = 0

            teacher_loss_mask = teacher_attention_mask.clone()
            student_loss_mask = student_attention_mask.clone()
            teacher_loss_mask[:, :len(bos_input_ids[0]) + len(question_input_ids[0])] = 0
            student_loss_mask[:, :len(bos_input_ids[0]) + len(question_input_ids[0]) + len(icot_input_ids[0])] = 0
            
            teacher_symbol_position = len(bos_input_ids[0]) + len(question_input_ids[0]) + len(cot_input_ids[0]) + len(symbol_input_ids[0]) - 1
            student_symbol_position = len(bos_input_ids[0]) + len(question_input_ids[0]) + len(icot_input_ids[0]) + len(symbol_input_ids[0]) - 1

            student_bot_position = len(bos_input_ids[0]) + len(question_input_ids[0])

            return {
                'teacher_input_ids' : teacher_input_ids,
                'student_input_ids' : student_input_ids,
                'teacher_attention_mask' : teacher_attention_mask,
                'student_attention_mask' : student_attention_mask,
                'teacher_loss_mask' : teacher_loss_mask,
                'student_loss_mask' : student_loss_mask,
                'teacher_symbol_position' : teacher_symbol_position,
                'student_symbol_position' : student_symbol_position,
                'student_bot_position' : student_bot_position
            }

        if self.split == 'test' :

            question = self.data[idx]['question']
            cot, answer =  self.data[idx]['answer'].split('\n#### ')
            
            bos_input_ids = self.tokenizer('<|endoftext|>', return_tensors = 'pt', add_special_tokens = False)['input_ids']
            question_input_ids = self.tokenizer(question, return_tensors = 'pt', add_special_tokens = False)['input_ids']
            icot_input_ids = self.tokenizer('<bot>' + '<|endoftext|>' * self.icot_length + '<eot>', return_tensors = 'pt', add_special_tokens = False)['input_ids']
            symbol_input_ids = self.tokenizer('The Answer is:', return_tensors = 'pt', add_special_tokens = False)['input_ids']
            
            input_ids = torch.cat([bos_input_ids, question_input_ids, icot_input_ids, symbol_input_ids], dim = 1)

            pad_input_ids = torch.cat([bos_input_ids for i in range(self.max_length - len(input_ids[0]))], dim = 1)
            input_ids = torch.cat([input_ids, pad_input_ids], dim = 1)

            attention_mask = torch.ones_like(input_ids)
            attention_mask[:, -len(pad_input_ids[0]):] = 0

            symbol_position = len(bos_input_ids[0]) + len(question_input_ids[0]) + len(icot_input_ids[0]) + len(symbol_input_ids[0]) - 1

            bot_position = len(bos_input_ids[0]) + len(question_input_ids[0])

            return {
                'answer' : answer,
                'input_ids' : input_ids,
                'attention_mask' : attention_mask,
                'bot_position' : bot_position,
                'symbol_position' : symbol_position
            }

class CODI_Model(nn.Module) :

    def __init__(self, model_path, icot_length = 6, alpha = 1, beta = 1, gamma = 1, max_length = 256) :
        
        super(CODI_Model, self).__init__()

        self.max_length = max_length
        self.icot_length = icot_length
        self.tokenizer = AutoTokenizer.from_pretrained('gpt2')
        self.tokenizer.add_special_tokens({'additional_special_tokens': ['<bot>', '<eot>']})
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained('gpt2', torch_dtype = torch.bfloat16)
        self.model.resize_token_embeddings(len(self.tokenizer), mean_resizing = True)
        config = LoraConfig(
            r = 128,
            lora_alpha = 32,
            target_modules = ['c_attn'],
            lora_dropout = 0.1,
            bias = 'none',
            task_type = 'CAUSAL_LM',
            fan_in_fan_out = True
        )
        self.model = get_peft_model(self.model, config).to('cuda')
        self.proj = nn.Sequential(
            nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size, dtype = torch.bfloat16),
            nn.GELU(),
            nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size, dtype = torch.bfloat16),
            nn.LayerNorm(self.model.config.hidden_size, dtype = torch.bfloat16)
        ).to('cuda')
        
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.ce_loss = nn.CrossEntropyLoss(reduction = 'none')
        self.l1_loss = nn.SmoothL1Loss()
    
    def forward(self, inputs) :

        teacher_input_ids = inputs['teacher_input_ids'].to('cuda')
        teacher_inputs_embeds = self.model.get_input_embeddings()(teacher_input_ids).squeeze(dim = 1)
        teacher_attention_mask = inputs['teacher_attention_mask'].to('cuda')
        teacher_loss_mask = inputs['teacher_loss_mask'].to('cuda')
        teacher_labels = torch.roll(teacher_input_ids, shifts = -1, dims = 2).to('cuda')
        teacher_labels[:, :, -1] = self.tokenizer.eos_token_id
        teacher_outputs = self.model(inputs_embeds = teacher_inputs_embeds, attention_mask = teacher_attention_mask, output_hidden_states = True)
        teacher_logits = teacher_outputs['logits']
        teacher_hidden_states = teacher_outputs['hidden_states']
        teacher_logits = teacher_logits.view(-1, teacher_logits.shape[-1])
        teacher_labels = teacher_labels.view(-1)
        teacher_loss_mask = teacher_loss_mask.view(-1)
        teacher_loss = self.ce_loss(teacher_logits, teacher_labels)
        teacher_loss = torch.mean(teacher_loss[teacher_loss_mask.bool()])

        student_input_ids = inputs['student_input_ids'].to('cuda')
        student_inputs_embeds = self.model.get_input_embeddings()(student_input_ids).squeeze(dim = 1)
        student_bot_position = inputs['student_bot_position'].to('cuda')
        student_attention_mask = inputs['student_attention_mask'].to('cuda')
        student_loss_mask = inputs['student_loss_mask'].to('cuda')
        student_labels = torch.roll(student_input_ids, shifts = -1, dims = 2).to('cuda')
        student_labels[:, :, -1] = self.tokenizer.eos_token_id
        for i in range(self.icot_length) :
            hidden_states = self.model(inputs_embeds = student_inputs_embeds, attention_mask = student_attention_mask, output_hidden_states = True).hidden_states[-1]
            for j in range(len(student_bot_position)) :
                student_inputs_embeds[j, student_bot_position[j] + 1] = self.proj(hidden_states[j, student_bot_position[j]])
                student_bot_position[j] += 1
        student_outputs = self.model(inputs_embeds = student_inputs_embeds, attention_mask = student_attention_mask, output_hidden_states = True)
        student_logits = student_outputs['logits']
        student_hidden_states = student_outputs['hidden_states']
        student_logits = student_logits.view(-1, student_logits.shape[-1])
        student_labels = student_labels.view(-1)
        student_loss_mask = student_loss_mask.view(-1)
        student_loss = self.ce_loss(student_logits, student_labels)
        student_loss = torch.mean(student_loss[student_loss_mask.bool()])

        teacher_symbol_positions = inputs['teacher_symbol_position']
        student_symbol_positions = inputs['student_symbol_position']
        
        distill_loss = 0
        for i in range(len(teacher_hidden_states) - 1) :
            teacher_distill_hidden_states = []
            student_distill_hidden_states = []
            for j in range(len(teacher_hidden_states[i])) :
                teacher_distill_hidden_states.append(teacher_hidden_states[i][j, teacher_symbol_positions[j]])
                student_distill_hidden_states.append(student_hidden_states[i][j, student_symbol_positions[j]])
            teacher_distill_hidden_states = torch.stack(teacher_distill_hidden_states)
            student_distill_hidden_states = torch.stack(student_distill_hidden_states)
            distill_loss += self.l1_loss(teacher_distill_hidden_states, student_distill_hidden_states) / torch.std(teacher_distill_hidden_states)
        distill_loss = distill_loss / (len(teacher_hidden_states) - 1)

        loss = self.alpha * teacher_loss + self.beta * student_loss + self.gamma * distill_loss

        return {
            'teacher_loss' : teacher_loss,
            'student_loss' : student_loss,
            'distill_loss' : distill_loss,
            'loss' : loss
        }
    
    def test(self, inputs) :

        true_answers = inputs['answer']
        input_ids = inputs['input_ids'].to('cuda')
        inputs_embeds = self.model.get_input_embeddings()(input_ids).squeeze(dim = 1)
        bot_position = inputs['bot_position'].to('cuda')
        attention_mask = inputs['attention_mask'].to('cuda')
        self.model.eval()
        for i in range(self.icot_length) :
            hidden_states = self.model(inputs_embeds = inputs_embeds, attention_mask = attention_mask, output_hidden_states = True).hidden_states[-1]
            for j in range(len(bot_position)) :
                inputs_embeds[j, bot_position[j] + 1] = self.proj(hidden_states[j, bot_position[j]])
                bot_position[j] += 1
        symbol_positions = inputs['symbol_position']
        results = []
        eos_reached = [False] * len(symbol_positions)
        while torch.min(symbol_positions).item() < self.max_length - 1 :
            logits = self.model(inputs_embeds = inputs_embeds, attention_mask = attention_mask).logits
            next_ids = []
            for j in range(len(symbol_positions)) :
                if eos_reached[j] :
                    next_ids.append(self.tokenizer.eos_token_id)
                elif symbol_positions[j] < self.max_length - 1 :
                    next_id = torch.argmax(logits[j, symbol_positions[j]])
                    next_ids.append(next_id.item())
                    next_embeds = self.model.get_input_embeddings()(next_id)
                    symbol_positions[j] += 1
                    attention_mask[j, 0, symbol_positions[j]] = 1
                    inputs_embeds[j, symbol_positions[j]] = next_embeds
                else :
                    next_ids.append(self.tokenizer.eos_token_id)
                    eos_reached[j] = True
            results.append(next_ids)
            if next_ids == [self.tokenizer.eos_token_id for i in range(len(symbol_positions))] :
                break
        results = torch.tensor(results).transpose(0, 1).tolist()
        answers = []
        for result in results :
            answers.append(self.tokenizer.decode(result, skip_special_tokens = True))
        return {
            'true_answers' : true_answers,
            'answers' : answers
        }

def CODI_train(model = 'gpt2', train_data_path = '../data/train.txt', test_data_path = '../data/test.parquet', epochs = 40, batch_size = 64, gradient_accumulation_steps = 2, lr = 3e-3, weight_decay = 1, warmup_rate = 0.03, test_steps = 3005, save_steps = 3005) :
    
    model = CODI_Model(model)

    train_data = CODI_Dataset(train_data_path, model.tokenizer)
    test_data = CODI_Dataset(test_data_path, model.tokenizer, split = 'test')
    train_data_loader = DataLoader(train_data, batch_size = batch_size, shuffle = True)
    test_data_loader = DataLoader(test_data, batch_size = batch_size, shuffle = False)

    optimizer = optim.AdamW(model.parameters(), lr = lr, weight_decay = weight_decay)
    total_steps = len(train_data_loader) * epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps = int(warmup_rate * total_steps), num_training_steps = total_steps)

    model.train()

    accumulated_steps = 0
    loss_list = []
    for epoch in range(epochs) :
        for batch in train_data_loader :
            loss = model(batch)['loss'] / gradient_accumulation_steps
            loss.backward()
            loss_list.append(loss.item())
            accumulated_steps += 1
            if accumulated_steps % gradient_accumulation_steps == 0 :
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                print(f'epoch : {epoch}, step : {accumulated_steps / gradient_accumulation_steps}, loss : {sum(loss_list)}')
                loss_list = []
            if (accumulated_steps / gradient_accumulation_steps) % test_steps == 0 :
                test_result = CODI_test(model, test_data_loader)
                with open(f'../result/test/{int((accumulated_steps / gradient_accumulation_steps) / test_steps)}.json', 'a') as f :
                    json.dump(test_result, f)
            if (accumulated_steps / gradient_accumulation_steps) % save_steps == 0 : 
                torch.save(model.state_dict(), f'../result/ckpt/{int((accumulated_steps / gradient_accumulation_steps) / save_steps)}.pth')
    torch.save(model.state_dict(), '../result/ckpt/final.pth')

def CODI_test(model, test_data_loader) :

    model.eval()
    answers = []
    true_answers = []
    for batch in tqdm(test_data_loader) :
        result = model.test(batch)
        answers = answers + result['answers']
        true_answers = true_answers + result['true_answers']
    return {
        'answers' : answers,
        'true_answers' : true_answers
    }

if __name__ == '__main__' :
    
    # 训练
    CODI_train()

    # 测试
    # num = 40
    # model = CODI_Model('gpt2')
    # model.load_state_dict(torch.load(f'../result/ckpt/{num}.pth'))
    # test_data = CODI_Dataset('../data/test.parquet', model.tokenizer, split = 'test')
    # test_data_loader = DataLoader(test_data, batch_size = 64, shuffle = False)
    # result = CODI_test(model, test_data_loader)
    # print(result)
