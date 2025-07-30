# Controllable and Stealthy Shilling Attacks via Dispersive Latent Diffusion (DLDA)

Recommender systems (RSs) are now fundamental to online platforms, but their dependence on user-contributed data leaves them vulnerable to shilling attacks, where adversaries inject fake users to manipulate item rankings. Despite extensive study, most existing attack models fail to meet two critical objectives simultaneously: achieving strong adversarial promotion of target items while maintaining realistic behavior to evade detection. As a result, the true severity of shilling threats that reconcile the two objectives remains underappreciated.
To expose this overlooked vulnerability, we present DLDA, a diffusion-based attack framework that generates fake users who are both highly effective at promoting target items and highly stealthy in mimicking natural user behavior. Specifically, DLDA employs a conditional latent diffusion process to iteratively construct fake interaction profiles guided by interaction-level preferences, while a dispersive regularization mechanism enhances behavioral variability to reduce detectable anomalies without sacrificing control over attack targets.
Experiments across three real-world datasets and five popular RS models show that DLDA consistently outperforms prior methods, achieving stronger item promotion while remaining harder to detect. These results highlight that modern RSs are more vulnerable than previously recognized, underscoring the urgent need for more robust defenses.

![](Figure/overview.png "The overview of DLDA")

## Requirements
```plaintext
pandas==1.3.5
numba==0.60.0
numpy==1.24.2
scipy==1.13.1
torch==2.1.0
