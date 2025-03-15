### 对比之下，发现可能不需要调整 support_masks 的尺寸，使其与 support_imgs 的尺寸匹配
```
support_masks = torch.stack(support_masks)
# support_masks_tmp = []
# for smask in support_masks:
#     smask = F.interpolate(smask.unsqueeze(0).unsqueeze(0).float(), support_imgs.size()[-2:],
#                           mode='nearest').squeeze()
#     support_masks_tmp.append(smask)
# support_masks = torch.stack(support_masks_tmp)
```
当然，这个并不影响最终实验结果

### self.length_data_list 对于不一样的数据集取不一样的值，但是 600 真的是 isic 数据集的长度(2596)吗？这样读取数据集会不会导致数据集没有使用完？
据说不用完所有数据，对结果影响不大

### 将sample_tasks里面的class_sample改为class_sample+1

```
def sample_tasks(self):
    ......
    for idx in range(total_length):
        class_id = idx % len(self.class_ids)
        class_sample = self.categories[class_id]
        ......
        self.tasks.append((query_name, support_names, class_id + 1))
```

**class_id + 1** 是为了让类别索引从 1 开始，而不是 0。
是否改动取决于你的数据格式和需求，如果数据集类别编号从 1 开始，改动是合理的。

可以发现 deepglobe 处理数据集的代码中：
```
self.categories = ['1', '2', '3', '4', '5', '6']
self.class_ids = range(0, 6)
```
而在之后 start_eval_loop 里面：
```
data_loader.dataset.sample_tasks()
val_labels = list(range(1,7))
```
**因此，将 class_id + 1 是很有必要的，而且就是导致最后结果出错的原因！**