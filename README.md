# Pyramid-Swin-Transformers

<img src="/images/block.png" width="500" height="200" />
The **Pyramid Swin Transformer** addresses and augments the original Swin Transformer, primarily focusing on improving multi-scale and computational complexities inherent in vision Transformers.

### Core Challenges
- Existing Swin Transformers use window-based multi-head self-attention, leading to isolated windows.
- While shifted window-based multi-head self-attention bridges some gaps, connectivity between some windows remains sparse, especially in large-scale dimensions.

### Our Solution
- Introduced a feature pyramid component for object detection, with modifications to the Swin Transformer structure.
- Design breaks down the attention process into smaller blocks, promoting global connectivity.
- Blend of local and global image information captured, with attention spread across windows of different sizes.

### Benefits
- Enhanced comprehension and recognition of target objects.
- More robust information exchange between windows, enhancing model performance.
- A balance between local feature refinement and global context understanding.

<img src="/images/swin.png" width="500" height="200" />

We also tested our model on different tasks, all with very good results.


### Variants
Proposed multiple Pyramid Swin Transformer variants:
- **Pyramid Swin-S**: C = 96, layer = {3,2,2,1}
- **Pyramid Swin-R**: C = 96, layer = {4,3,2,2}
- **Pyramid Swin-B**: C = 128, layer = {4,3,3,2}
- **Pyramid Swin-L**: C = 192, layer = {4,3,3,2}

### Conclusion
Our Pyramid Swin Transformer is an evolved architecture, boosting efficiency in multi-task computer vision scenarios, and ensuring better information interaction between windows.



