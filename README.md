# 📦 Image Compression Project Results  
*Using WGAN & Linformer*

This report presents the visual and quantitative results of an image compression framework based on **Wasserstein GAN (WGAN)** and **Linformer** attention mechanisms. The approach focuses on maintaining high visual fidelity while ensuring efficient compression.

---

## 🖼️ Result 1: Input vs. Reconstructed Image
images
![Input vs Reconstructed Image](./images/input_output_image.png)

*Description:*  
This figure shows a side-by-side comparison of the original input image and the reconstructed image after compression and decompression using the proposed model. The reconstructed output retains key features, textures, and visual coherence, indicating the model's ability to preserve perceptual quality.

---

## 📉 Result 2: Training Loss Curves

![Loss Curves](./image2.png)

*Description:*  
The plot illustrates the training losses of the **Generator**, **Critic**, and **Encoder** over the training epochs.  
- The **Generator Loss** indicates the model's progress in fooling the critic.
- The **Critic Loss** reflects its ability to differentiate between real and fake images.
- The **Encoder Loss** helps maintain latent consistency during compression.

These curves show convergence behavior and provide insight into the training dynamics and model stability.

---

## 📈 Result 3: Evaluation Metrics – PSNR, LPIPS, SSIM

![PSNR, LPIPS, SSIM Metrics](./image3.png)

*Description:*  
The third plot presents quantitative metrics:
- **PSNR (Peak Signal-to-Noise Ratio)**: Measures reconstruction fidelity in terms of pixel similarity.
- **LPIPS (Learned Perceptual Image Patch Similarity)**: Captures perceptual similarity from deep features.
- **SSIM (Structural Similarity Index Measure)**: Evaluates structural integrity of reconstructed images.

Higher PSNR & SSIM and lower LPIPS scores across different test samples confirm the model's effectiveness in preserving perceptual and structural quality during compression.

---

## ✅ Conclusion

The experimental results demonstrate:

- Effective reconstruction of images with high visual quality.
- Stable training dynamics across all model components.
- Strong performance in standard evaluation metrics (PSNR, LPIPS, SSIM), showing that the model balances pixel accuracy and perceptual fidelity well.

This confirms the potential of integrating **WGAN** with **Linformer-based compression** for image compression tasks.

---

## 📁 Notes

Ensure the following files are in the same directory as this Markdown file:
- `image1.png` — Input vs. Reconstructed Image
- `image2.png` — Training Loss Curves
- `image3.png` — Evaluation Metrics

