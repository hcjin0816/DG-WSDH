import torch
def average_precision_patch(query, ranking):
    query = query
    scores = []
    j = 0
    index = 0
    for i in range(len(ranking)):
        index += 1
        if query == ranking[i]:
            j += 1
            score = j / index
            scores.append(score)

    if j != 0:
        return sum(scores) / len(scores)
    else:
        return 0

# train 的 hashcodes
patch_hash_codes_online = torch.load('')['hashcodes'] # save的hash codes 文件路径
wsi_bag_patch_label_online = torch.load('')['patch_label']  # save的hash codes 文件路径
ap_online = 0
for i in range(len(patch_hash_codes_online)):
    if len(patch_hash_codes_online[i]) != 0:
        patch_hash_codes_online[i] = torch.stack(patch_hash_codes_online[i]).view(-1, 64)
        wsi_bag_patch_label_online[i] = torch.tensor([i for j in wsi_bag_patch_label_online[i] for i in j])
        num_rows, num_cols = patch_hash_codes_online[i].size()
        random_indices = torch.randperm(num_rows)

        # 使用相同的随机索引重新排列两个 tensor 的行
        wsi_bag_patch_label_online[i] = wsi_bag_patch_label_online[i].view(-1, 1)[random_indices, :]
        patch_hash_codes_online[i] = patch_hash_codes_online[i][random_indices, :]
        query = patch_hash_codes_online[i] / torch.norm(patch_hash_codes_online[i], dim=-1,
                                                        keepdim=True)  # 方差归一化，即除以各自的模
        patch_labels_online = wsi_bag_patch_label_online[i].view(1, -1).cuda()
        patch_label = wsi_bag_patch_label_online[i].view(1, -1).tolist()[0]
        # 防止内存溢出，分批进行topk计算
        for j in range(int(num_rows/1000)):
            j = j+1
            temp_query = query[(j - 1) * 1000:(j * 1000)]
            temp_patch_label = patch_label[(j - 1) * 1000:(j * 1000)]
            if j == int(num_rows/1000):
                temp_query = query[((j-1)*1000):]
                temp_patch_label = patch_label[((j-1)*1000):]
            similarity = torch.mm(temp_query, query.t())  # 矩阵乘法
            max_values, max_indices = torch.topk(similarity, 50, dim=1, largest=True)  # (128, 50)
            ap_a_online = 0
            for z in range(similarity.shape[0]):
                # patch_labels_online中选择对应的patch_label
                label_ = torch.gather(patch_labels_online, 1, max_indices[z].view(1, -1)).tolist()[0]
                ap_a_online += average_precision_patch(temp_patch_label[z], label_)
            ap_online.append(ap_a_online / similarity.shape[0])
            print(sum(ap_online)/len(ap_online), i)

# test的hashcode
patch_hash_codes_online = torch.load('')['hashcodes']  # save的hash codes 文件路径
wsi_bag_patch_label_online = torch.load('')['patch_label']  # save的hash codes 文件路径
for i in range(len(patch_hash_codes_online)):
    if len(patch_hash_codes_online[i]) != 0:
        patch_hash_codes_online[i] = torch.stack(patch_hash_codes_online[i]).view(-1, 64)
        wsi_bag_patch_label_online[i] = torch.tensor([i for j in wsi_bag_patch_label_online[i] for i in j])
        num_rows, num_cols = patch_hash_codes_online[i].size()
        random_indices = torch.randperm(num_rows)

        # 使用相同的随机索引重新排列两个 tensor 的行
        wsi_bag_patch_label_online[i] = wsi_bag_patch_label_online[i].view(-1, 1)[random_indices, :]
        patch_hash_codes_online[i] = patch_hash_codes_online[i][random_indices, :]
        query = patch_hash_codes_online[i] / torch.norm(patch_hash_codes_online[i], dim=-1,
                                                        keepdim=True)
        patch_label = wsi_bag_patch_label_online[i].view(1, -1).tolist()[0]
        patch_labels_online = wsi_bag_patch_label_online[i].view(1, -1).cuda()
        for j in range(int(num_rows / 1000)):
            j = j + 1
            temp_query = query[(j - 1) * 1000:(j * 1000)]
            temp_patch_label = patch_label[(j - 1) * 1000:(j * 1000)]
            if j == int(num_rows / 1000):
                temp_query = query[((j - 1) * 1000):]
                print(temp_query.shape)
                temp_patch_label = patch_label[((j - 1) * 1000):]
            similarity = torch.mm(temp_query, query.t())  # 矩阵乘法
            max_values, max_indices = torch.topk(similarity, 50, dim=1, largest=True)  # (128, 50)
            ap_a_online = 0
            for z in range(similarity.shape[0]):
                label_ = torch.gather(patch_labels_online, 1, max_indices[z].view(1, -1)).tolist()[0]
                ap_a_online += average_precision_patch(temp_patch_label[z], label_)
            ap_online.append(ap_a_online / similarity.shape[0])
            print(sum(ap_online) / len(ap_online),i)
