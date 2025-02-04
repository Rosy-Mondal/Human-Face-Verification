Face verification is a challenging task, especially under difficult conditions such asvarying poses, lighting, facial expressions, and obstructions. This challenge becomeseven harder when relying on a single source of training data, which often fails torepresent the complexity of facial variations. To address this issue, we present a newapproach called GaussianFace, based on a method known as the DiscriminativeGaussian Process Latent Variable Model (DGPLVM).
Unlike traditional methods that depend on one training dataset, our model uses datafrom multiple sources to better handle the differences found in unknown or unseenscenarios. This ability to adapt to complex data distributions allows GaussianFace toeffectively capture the wide range of variations in human faces.
To make the model more effective at distinguishing between different faces, weimproved it by integrating an efficient version of Kernel Fisher Discriminant Analysis.
Additionally, we used a low-rank approximation technique to make the process of
prediction and inference faster.
We conducted extensive experiments to test our model, and the results show that it
performs exceptionally well at learning from diverse data and generalizing to new,
unseen environments. Most notably, our approach achieved a remarkable accuracy of98.52% on the challenging Labeled Faces in the Wild (LFW) benchmark, surpassinghuman-level performance (97.53%) in face verification for the first time.
