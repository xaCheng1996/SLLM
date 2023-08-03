import torch
import clip


def text_prompt(data, prompt_mode='enhance'):
    text_aug = [f"{{}}"]

    # text_aug_pre = [f"a photo of action {{}}, in other words {{}}",
    #                 f"a picture of action {{}}, which is a action of {{}}",
    #                 f"Human action of {{}}, {{}}",
    #                 f"{{}}, an action, {{}}",
    #                 f"{{}} this is an action, furthermore, {{}}",
    #                 f"{{}}, a video of action, that is {{}}",
    #                 f"Playing action of {{}}, in other words, {{}}",
    #                 f"{{}}, {{}}"]
    #
    # text_aug_sup = [
    #                 f"{{}} is playing a kind of action, {{}}",
    #                 f"{{}} is doing a kind of action, {{}}",
    #                 f"Can you recognize the action of {{}}, i.e., {{}}?",
    #                 f"{{}}, video classification of {{}}",
    #                 f"{{}}, A video of {{}}",
    #                 f"{{}}, The man is {{}}",
    #                 f"{{}}, The woman is {{}}"
    #             ]
    text_dict = {}

    if prompt_mode == 'normal':
        num_text_aug = len(text_aug)
        for ii, txt in enumerate(text_aug):
            text_dict[ii] = torch.cat([clip.tokenize(txt.format(c)) for i, c, d in data.classes])
    elif prompt_mode == 'enhance':
        num_text_aug = len(text_aug)
        for ii, txt in enumerate(text_aug):
            text_dict[ii] = torch.cat([clip.tokenize(txt.format(c + ',' + d)) for i, c, d in data.classes])
    else:
        print('Not support this text prompt:{}'.format(str(prompt_mode)))
        return

    classes = []
    for k, v in text_dict.items():
        classes.append(text_dict[k])
    classes = torch.cat(classes)
    return classes, num_text_aug, text_dict