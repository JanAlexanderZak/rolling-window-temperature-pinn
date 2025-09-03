# Rolling Window Temperature PINN

This project demonstrates two key techniques for integrating experimental data into Physics-Informed Neural Networks (PINNs) for improved accuracy and efficiency:

1.  **Rolling Window for Learning Rate:** Addresses the issue of premature learning rate reduction caused by the introduction of experimental loss terms. By using a rolling window approach, the learning rate adapts more effectively to the overall loss landscape, preventing stagnation during training.

2.  **Lookup Table for Temperature Updates:** Presents a high-performance alternative to interpolation methods for updating temperature values based on experimental data. The lookup table approach offers a significant speedup (10-100x) compared to interpolation, making it ideal for real-time or computationally intensive applications.
