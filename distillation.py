import torch.nn.functional as F


def cosine_distill(student_feat, teacher_feat):
    s = F.normalize(student_feat, dim=-1)
    t = F.normalize(teacher_feat, dim=-1)
    return (1 - (s * t).sum(dim=-1)).mean()