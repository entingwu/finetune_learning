import torch

def compute_loss(chosen_log_probs, rejected_log_probs, ref_chosen_log_probs, ref_rejected_log_probs):
  """
  DPO的损失函数：
  chosen_log_probs: shape [batch_size, ]
  rejected_log_probs: shape [batch_size, ]
  ref_chosen_log_probs: shape [batch_size, ]
  ref_rejected_log_probs: shape [batch_size, ]
  beta: 超参数
  """

  margin = chosen_log_probs - rejected_log_probs - (ref_chosen_log_probs - ref_rejected_log_probs)
  # loss: shape [batch_size, ]
  loss = - torch.nn.functional.logsigmoid(beta * margin)

  return loss.mean() # 求平均


def compute_log_probs(logits, labels, assistant_mask):
  """
  计算模型输出label当中回答的对数概率
  """
  # 1. 获取到log_probs, shape: [batch_size, num_tokens, vocab_size]
  log_probs = torch.log_softmax(logits, dim=-1)

  # 2. 需要从log_probs里面找到，基于前n个token， 输出答案token的具体的对数概率
  # [batch_size, num_tokens]
  label_token_log_prob = torch.gather(input=log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

  # 3. 对负对数概率，进行掩码，让非assistant answer部分，置为0，assistant answer保留原值。
  # [batch_size, num_tokens]
  masked_label_token_log_prob  = assistant_mask * label_token_log_prob

  # 4. 对对数概率相加，获取模型输出整个答案的对数概率
  # shape：普通的标量
  log_prob.sum(dim = -1)

  return average_loss