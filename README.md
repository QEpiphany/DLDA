# Controllable and Stealthy Shilling Attacks via Dispersive Latent Diffusion (DLDA)

Recommender systems (RSs) are now fundamental to online platforms, but their dependence on user-contributed data leaves them vulnerable to shilling attacks that can manipulate item rankings by injecting fake users. Although widely studied, most existing attack models fail to meet two critical objectives simultaneously: achieving strong adversarial promotion of target items while maintaining realistic behavior to evade detection. As a result, the true severity of shilling threats that manage to reconcile the two objectives remains underappreciated. To expose this overlooked vulnerability, we present DLDA, a diffusion-based attack framework that can generate highly effective yet indistinguishable fake users by enabling fine-grained control over target promotion. Specifically, DLDA operates in a pre-aligned collaborative embedding space, where it employs a conditional latent diffusion process to iteratively synthesize fake user profiles with precise target item control. To evade detection, DLDA introduces a dispersive regularization mechanism that promotes variability and realism in generated behavioral patterns. Extensive experiments on three real-world datasets and five popular RS models demonstrate that, compared to prior attacks, DLDA consistently achieves stronger item promotion while remaining harder to detect. These results highlight that modern RSs are more vulnerable than previously recognized, underscoring the urgent need for more robust defenses.

![](Figure/overview.png "The overview of DLDA")

## Requirements
```plaintext
pandas==1.3.5
numba==0.60.0
numpy==1.24.2
scipy==1.13.1
torch==2.1.0
