\subsection{Reviewer 1}

SCORE: -1 (weak reject)  
----- TEXT:  
### Summary  
This paper introduces a binarization technique for a particular kind of neural network, which enables to replace some layers by unions or intersections of binary MM operators.  
While relevant to DGMM, I found that this paper is hard to read and lacks many details, preventing me from really understanding the proposed approach.  
  
### Paper overview  
This paper introduces:  
1. an architecture with morphological neurons and layers which can be trained using standard techniques, using float weights;  
2. a technique to binarize the structuring elements used in these networks.  
  
Regarding the architecture: a single neuron consists in a convolution between an input and a structuring element, followed by a bias subtraction (bias should be 1 for dilation, |S| for erosion). Combining these neurons, layers can be formed and merged using intersection or union operations.  
  
Regarding the binarization technique: using a definition of an "almost binary image", the authors derive the properties that the weights of a SE should exhibit to be directly replaceable by a morphological operator. In case such properties are not observed, an approximate binarization is proposed to recover the binary SE which best approximates the target morphological operation.  
  
The authors tested the capability of such layers to capture simple structuring elements on small datasets.  
For each experiment, there are only a handful of layers involved (only 1 for the first experiments).  
The authors showed their architecture was capable to learn correct disk, stick and cross SEs with the following setup: training input is an original (small) image, and target output is the original image transform with a dilation, or erosion, or opening, or closing by the target structuring element.  
Experiments involving several layers and white/black tophat were not really successful.  
The binarization strategy seems to work on simple cases (dilation, erosion, opening, or closing).  
  
### Strengths  
- The scope of this paper is relevant to the DGMM community.  
- The source code of this paper is publicly available on GitHub.  
- The binarization approach seems viable on simple cases.  
- I found the concept of "almost binary image" original.  
  
### Weaknesses  
1. To me, the claim of the paper is unclear: I understand this paper as a neural net. binarization technique for a special case of neural net. architecture. In particular, as there is no support for any "explainability" of the resulting architecture, I do not think we should advertise the MM networks as more explainable than regular ones.  *=> Explainability comes with the morphological operation that is more explanable than the convolution.*
2. I found section 2.2 very hard to read as it introduces a lot of concepts but lacks some critical details for my understanding:  
- How do you ensure that the weights are > 0?  *=> Thm 1 shows that the weights are positive. Then we enforce it in the BISE expression with a softplus.*
- How do you set the values of u and v as a criterion for an "almost binary image"?  *The input has u, v = 0, 1. Then, if the BiSE is activated, the output is almost binary and we know the u, v depending on the weights and bias.*
- The consideration of negative weights in eq. (12) and (13) are very confusing at this point; is this really necessary to consider anti-{erosion,dilation} at this point?  *=> The consideration of negative is used to justify a posteriori that the weights can be set to positive. Anti erosion and anti dilation are necessary to learn a wide panel of morphological operators.*
- Given the candidate SE is learned a set of float values, how do you convert this set of float values to a binary structuring element (noted a set) in eq. (12) and (13)?  *=> Thm 1 allows to check if the current state of the BiSE neuron can be replaced by a morphological operation with the Sel S. The equation is respected for at most one S. If this S exists, we can find it by applying the threshold of proposition 2. If it does not exist, we apply an approximate binarization explained in subsubsection Binarization.*
3. The experimental section lacks some important details:  
- There is little detail about the actual architecture used. I had to check the preprint to understand them!  *=> This should be clarified. The architecture is similar to the morphological operations.*
- The experiments do not report which layers were actually "activated", i.e. binarized using the direct proposed formulation, versus the approximate one. Is the drop in performance for some results due to the use of approximation?  *=> We chose to only show the "binarization" operation, described in section 2.2. We believe that the table would contain too much information if we added the activated part. The answer to the question is yes, if there is a drop in performance, this means that the bise was not activated.*
- It is said that binarization would improve network speed, but there is no support for this claim, either from a theoretical study of the operations needed, nor from a measure on some particular hardware.  *=> add a reference for the binarization community*
- It would be really nice to see some evaluation of the binarized architecture on a test set for the axSpA dataset, or any real task, to prove the usability of the binarized network.  *=> The DICE is reported on the test set for the axSpA dataset.*
4. I also found the sections 3 and 4 badly organized:  
- I strongly suggest separating the experiment on the axSpA dataset to be moved to a separate subsection (merge 3.2 and 4.3) to isolate this "real case" from theoretical, preliminary experiments.  *=> Thank you for your input.*
- I also strongly suggest to use colored structured elements for float values (this is what the colormap is for) and keep black and white colors for binary SEs. It was very confusing for me at first sight. *=> Thank you for your input we will change that.*


### Fixable weakness
4. I also found the sections 3 and 4 badly organized:  
- I strongly suggest separating the experiment on the axSpA dataset to be moved to a separate subsection (merge 3.2 and 4.3) to isolate this "real case" from theoretical, preliminary experiments.  *=> Thank you for your input.*
- I also strongly suggest to use colored structured elements for float values (this is what the colormap is for) and keep black and white colors for binary SEs. It was very confusing for me at first sight. *=> Thank you for your input we will change that.*

### Unanswered weaknesses
3. The experimental section lacks some important details:  
- There is little detail about the actual architecture used. I had to check the preprint to understand them!  *=> This should be clarified. The architecture is similar to the morphological operations.*
- The experiments do not report which layers were actually "activated", i.e. binarized using the direct proposed formulation, versus the approximate one. Is the drop in performance for some results due to the use of approximation?  *=> We chose to only show the "binarization" operation, described in section 2.2. We believe that the table would contain too much information if we added the activated part. The answer to the question is yes, if there is a drop in performance, this means that the bise was not activated.*
- It is said that binarization would improve network speed, but there is no support for this claim, either from a theoretical study of the operations needed, nor from a measure on some particular hardware.  *=> add a reference for the binarization community*

- The consideration of negative weights in eq. (12) and (13) are very confusing at this point; is this really necessary to consider anti-{erosion,dilation} at this point?  *=> The consideration of negative is used to justify a posteriori that the weights can be set to positive. Anti erosion and anti dilation are necessary to learn a wide panel of morphological operators.*

\subsection{Reviewer 2}

SCORE: -1 (weak reject)  
----- TEXT:  
This paper studies an interesting problem for the mathematical morphology community which is how to learn binary morphological operators following the paradigm of learning from data via gradient descent. The authors study a formulation of flat dilation and erosion by a linear convolution and a threshold. The difficulty of approximating the threshold is contoured by a soft approximation. This reviewer thinks that the motivation and proposed ideas of the paper are interesting. However, in the current state the paper can not be accepted.  
  
a) the paper has a large overlap with the reference [2]. It should be nice that the paper clarifies that in the introduction, indicating a list of contributions. That could help to understand the "Multi Channel" in the title, which is never mentioned in the paper.  *=>  Thank you, we should add it to the paper. The preprint is not a peer-reviewed document. The proposed paper is a stand-alone paper, we just cited the preprint as it is available online. The multi-channel part comes with the ability to learn multiple morphological operators per layer.*
  
b) There are many sentences that are not correctly justified. Here you have some examples (non-exhaustive list):  
  
- Deep learning have been comparatively little studied from the theoretical point of view. -> Even if I agree with the authors about the fact that MM has solid theoretical foundations, the comparison about studying from a theoretical point of view seems to me as a personal point of view.  *=> Quote the paper of Jesus*
- We design a single neuron that replaces the convolution operation -> For me understanding, your proposition is solving another kind of problem and it is not looking for an alternative method for CNNs. Indeed you are using convolution to approximate your operators.  *=> The neuron uses the convolution, but is not exactly the same. It also has equivalence property. We can say that we enhance the CNN with explanability and morphological equivalence.*
- we will explain that latter -> This is not formal English. You should indicate what subsection you explain.  *=> Thank you for your input. This will be changed.*
- For each result, see the code for the hyperparameters used. -> This reviewer thinks that sending the readers to a code is not a good practice in scientific articles.  *=> Ok, remove the hyperparameters from the paper.*
  
  
c) Some choices without justification:  
- Do you have any justification for choosing fˆ{+} as softplus, or any equivariant activation type Relu could be used? (Softplus is indeed an approximation of Relu). *=> Any function that maps W to $\R^+$ can be used. We chose one among others.*
- I would like that in the experimental section authors replace fˆ{+} for fˆ{+}+.5 without any justification (only indicating that it helps during training).  *=> This could be more detailed. Sometimes during training, the bias could go really close to 0, resulting in instabilities.*
- Do you have any justification for choosing e(x)=1/2 tanh(x) + 1/2? Or any function mapping values to [0,1] could be used? *=> explained in the paper* 
- Do you have any justification for choosing Uniform Initialization for weights? p=0? And biases at 0.63?  *=> None at this stage. As said in the paper, the initialization is left for future work.*
The next phrase after initialization is: "Could we find other unbiased methods toward a particular operation or Sel ? This is left for further research", that looks more as a future work phrase. *=> Indeed it is.* 
- In the Table 1, DICE>.8 is highlighted, why? I do not see what extra information is given by this threshold.  *=> The threshold is arbitrary selected to highlight success. Even if 1 is not reached, a good dice can still be considered a success, even if not a perfect one.*
  
  
d) Some mistakes:  
There is something not clear in the document:  
-> In theorem 1, authors say p\in \real a scaling factor, but in (12),(13) $p = +\infty$ (the third term of erosion).  *=> Thank you, it will be fixed.*
-> Table 2 does not exist.  *=> Thank you, it was indeed figure 2.*
-> Fig 5 b) A kernel 21x21x2 can not give an output of dimension one.  *=> Thank you, it is indeed 21x21x1. It will be fixed.*
- Performance can decrease (see disk erosion on inverted MNIST) or increase (see disk erosion on MNIST) -> In these cases all the DICE results are one.?  *=> check pkoi j'ai dit ça*
-We cannot solve classification tasks with our methods -> What is the problem of using strides in the current formulation to increase a filter size reduction on the operator?  *=> Thank you for your idea, it is a possibility that is left for future research. Expliquer pk c'est pas straightforward?*
  
e) Missing reference:  
  
For the introduction about MM and DL, this reviewer would like to recommend the following reference, which should only be included if the authors consider it pertinent.  *=> Thank you for your input, we will look into it.*
  
Velasco-Forero, S., Pagès, R., & Angulo, J. (2022). Learnable Empirical Mode Decomposition based on Mathematical Morphology. SIAM Journal on Imaging Sciences, 15(1), 23-44.  
  
f) Typo mistakes:  
  
MM can se used to construct -> MM can be used to construct  
We introduces the Binary Morphological Neural Network -> We introduce the Binary Morphological Neural Network  
with pixels value either close to 0 or close to 1 -> with pixel value either close to 0 or close to 1  
Binarizing a BiSEL is equivalent to binarizing each of its element -> Binarizing a BiSEL is equivalent to binarizing each of its elements.  
we can only learn predeternined -> we can only learn predetermined

### Fixable weakness
a) the paper has a large overlap with the reference [2]. It should be nice that the paper clarifies that in the introduction, indicating a list of contributions. That could help to understand the "Multi Channel" in the title, which is never mentioned in the paper.  *=>  Thank you, we should add it to the paper. The preprint is not a peer-reviewed document. The proposed paper is a stand-alone paper, we just cited the preprint as it is available online. The multi-channel part comes with the ability to learn multiple morphological operators per layer.*
- we will explain that latter -> This is not formal English. You should indicate what subsection you explain.  *=> Thank you for your input. This will be changed.*
- For each result, see the code for the hyperparameters used. -> This reviewer thinks that sending the readers to a code is not a good practice in scientific articles.  *=> Ok, remove the hyperparameters from the paper.*

There is something not clear in the document:  
-> In theorem 1, authors say p\in \real a scaling factor, but in (12),(13) $p = +\infty$ (the third term of erosion).  *=> Thank you, it will be fixed.*
-> Table 2 does not exist.  *=> Thank you, it was indeed figure 2.*
-> Fig 5 b) A kernel 21x21x2 can not give an output of dimension one.  *=> Thank you, it is indeed 21x21x1. It will be fixed.*


MM can se used to construct -> MM can be used to construct  
We introduces the Binary Morphological Neural Network -> We introduce the Binary Morphological Neural Network  
with pixels value either close to 0 or close to 1 -> with pixel value either close to 0 or close to 1  
Binarizing a BiSEL is equivalent to binarizing each of its element -> Binarizing a BiSEL is equivalent to binarizing each of its elements.  
we can only learn predeternined -> we can only learn predetermined


### Unanswered weakness
- Deep learning have been comparatively little studied from the theoretical point of view. -> Even if I agree with the authors about the fact that MM has solid theoretical foundations, the comparison about studying from a theoretical point of view seems to me as a personal point of view.  *=> Quote the paper of Jesus*
- We design a single neuron that replaces the convolution operation -> For me understanding, your proposition is solving another kind of problem and it is not looking for an alternative method for CNNs. Indeed you are using convolution to approximate your operators.  *=> The neuron uses the convolution, but is not exactly the same. It also has equivalence property. We can say that we enhance the CNN with explanability and morphological equivalence.*
- Performance can decrease (see disk erosion on inverted MNIST) or increase (see disk erosion on MNIST) -> In these cases all the DICE results are one.?  *=> check pkoi j'ai dit ça*
- We cannot solve classification tasks with our methods -> What is the problem of using strides in the current formulation to increase a filter size reduction on the operator?  *=> Thank you for your idea, it is a possibility that is left for future research. Expliquer pk c'est pas straightforward?*


\subsection{Reviewer 3}

SCORE: -2 (reject)  
----- TEXT:  
This paper presents a novel type of morphological network operating with binary images and whose weights are also binary. The proposed binary structuring element neuron is able to learn a structuring element and to approximate a morphological erosion and dilation. Authors also propose a way for the morphological binary filters to be combined through union and intersection, allowing the network at the end of the day to learn more advanced operations such as opening/closing and top-hats.  
The article is clearly in the scope of DGMM as the integration of morphological operations within neural network architectures is currently a (hot) topic of interest for the community.  
  
The paper presents several interesting and promising ideas such as the notion of almost binary images and the ability to learn/recombine filter outputs with binary intersection and union operations. However, both the style and content suffer from many flaws, listed below:  
1. the paper is well positioned with respect to the other recent works concerned with the learning of a structuring element/morphological operation. However, its positioning with respect to binary neural network is much less clear: 3 references (numbered [6], [7] and [15]) are given in the paper to motivate the use of binary networks, but the state-of-the-art related to binary neural network is much richer and this is not reflected in the article. Here are some references that could be added for instance:  *=> We thank you for these references. The scope of the paper is not to compare our method to binarized neural network state of the art.*
- Rastegari, M., Ordonez, V., Redmon, J., & Farhadi, A. (2016, October). Xnor-net: Imagenet classification using binary convolutional neural networks. In European conference on computer vision (pp. 525-542). Springer, Cham.  
- Juefei-Xu, F., Naresh Boddeti, V., & Savvides, M. (2017). Local binary convolutional neural networks. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 19-28).  
- Lin, X., Zhao, C., & Pan, W. (2017). Towards accurate binary convolutional neural network. Advances in neural information processing systems, 30.  
2. In the definition of a BiSE neuron, it is said that the weights and bias are enforced to be positive. In theorem 1, two expressions (eq (12) and (13)) are given: if one of these is true, then the BiSE neuron is said to be activated, and a consequence of this is that weights and bias are positive. However, both equations (12) and (13) feature a sum on negative weights (last sum of the right-hand side). This is contradictory and unclear.  *=> This must be clarified in the paper. At first, we consider every set of weights. Then we find out that if the inequality is respected, then the set of weights is positive. Thus, we enforce positivity of the weights in the BiSE definition because only the positive weights are of interest. This is a justification of the positivity of weights inside the BISE neuron.*
3. The same comment applies to proposition 2: the BiSE is assumed to be activated, thus its weights are positive. However, equation (14) shows again a sum on negative weights.  *=> To improve clarity, we could only consider positive weights. But this property is more general with all kind weights, if we do not choose to enforce positivity.*
4. The same goes for Theorem 2/LUI. The weights beta_k are positive as a consequence of the activation of the LUI, but the last sum of the right-hand side term of equation (21) and (22) holds on negative weights. For 2. 3. and 4., if there is a theoretical justification to these seemingly contradictory facts, it should be clearly stated  *=> Same as 2.*
5. Theorems 1 and 2 cannot be called theorems as it stands as they both lack a formal proof of their validity.  *=> Proof exists but we do not have enough room.*
6. The role of the parameter p is unclear. It is said that its sign determines whether a complementation is applied or not, but is it a trainable parameter (as the p of the PConv layer) or is it set by the user (as said in the preprint where p is set to 4)?  *=> This should be clarified, p is a learnable parameter.*
7. I understand the necessity of defining a dissimilarity function for the binarization of a non-activated BiSE neuron. A dissimilarity is proposed by equation (17), what is the motivation behind this particular definition? It is said to be a first choice, does it mean that other dissimilarities have been implemented and compared?  *=> This is an interesting topic to discuss. We could clarify in the paper our choice for this dissimilarity function. The first choice is to compute how far away the BiSE parameters are from the activation. The set of activated parameters is a convex set, and this is basically the distance of the bias to this convex set. As said in the future work, finding better dissimilarity function is under study.*
8. The binarization of a non-activated LUI is also unclear. It is said that the same method described for the BiSE is applied, does it mean that it also relies on the definition of a dissimilarity function? If yes, how is defined this dissimilarity?  *=> The LUI is basically a BiSE with enforced structure. All properties of BiSE apply to LUI.*
9. I understand that when the BiSE and LUI are activated, it means that the conducted operation can be well approximated by a binary operation. When they are not activated however, the binarization is "forced" and this is rather intriguing. What would be the cause of a layer not activated? Is it because the learning has been stopped too early? Or is it simply because there is no binary operation to be approximated here? In which case, I assume that forcing a binarization would yield some errors/inconsistencies pretty much like using a standard convolutional layer that has not yet converged? The effects of forcing a binarization on a non-activated BiSE/LUI layer should be further presented and discussed, as it is a key point of the proposed method.  *=> Thank you for your discussion. We agree that it should be more discussed. If the BiSE is activated, it can be exactly replaced by the morphological operator, it is not an approximation. Some activations do not happen because nothing enforces that it should. Sometimes we reach perfect result without needing the activation to be complete.*
10. Fig 2 is referenced as table 2 in the text, and is referenced later than figure 3.  *=> Will be fixed.*
11. In the experiment part, why SEs used for erosion and dilation do not have the same size as the SEs used for the other morphological operations?  *=> Otherwise the erosion would almost delete the MNIST digits.*
12. Authors should not refer the reader to the code in the github repository for something as fundamental as the hyperparameters setting. These should be clearly presented in the main paper, or at least in a written supplementary materials document. Same goes for the learned SEs  *=> We will remove the reference to github repo.*
13. Something appears unclear to me with the results presented in table 1: it is said that SEs are well recovered for all scenarios except the opening with the inverted MNIST, for which the DICE score after binarization is below 0.9. However, the score presented for the closing with the stick on the Diskorect dataset are worse.  *=> We consider that dice score below 0.8 is a failure.*
14. Authors refute the (plausible) hypothesis that dual operators are learnt similarly on dual datasets, but without any further explanation or attempt to explain why  *=> The why is observed in similar works, add citation. It is a mystery that is left for future work, as stated in paper.*
15. Authors say that they have compared their results with those of LMorph and SMorph layers, but they do not show any of those comparisons  *=> We only explain the results we have on LMorph and SMorph*.
16. Figure 4 appears unclear to me: the upper left weight matrix in Fig 4a has 2 high entries (white pixels), but only 1 remain after binarization in Fig 4b. Also, the colormap used is rather counterintuitive: black/white would be much more suited to display binary weights.  *=> Thank you for advice for colormap.*
17. Results for architecture 3 in the AxSpA experiments are not reported. Furthermore, the results would be more readable if presented in a table rather than in the text. *=> This could be clarified: the network did not really move from its initial state and all the outputs are constant.*


### Fixable weakness
5. Theorems 1 and 2 cannot be called theorems as it stands as they both lack a formal proof of their validity.  *=> Proof exists but we do not have enough room.*
6. The role of the parameter p is unclear. It is said that its sign determines whether a complementation is applied or not, but is it a trainable parameter (as the p of the PConv layer) or is it set by the user (as said in the preprint where p is set to 4)?  *=> This should be clarified, p is a learnable parameter.*
8. The binarization of a non-activated LUI is also unclear. It is said that the same method described for the BiSE is applied, does it mean that it also relies on the definition of a dissimilarity function? If yes, how is defined this dissimilarity?  *=> The LUI is basically a BiSE with enforced structure. All properties of BiSE apply to LUI.*
10. Fig 2 is referenced as table 2 in the text, and is referenced later than figure 3.  *=> Will be fixed.*
12. Authors should not refer the reader to the code in the github repository for something as fundamental as the hyperparameters setting. These should be clearly presented in the main paper, or at least in a written supplementary materials document. Same goes for the learned SEs  *=> We will remove the reference to github repo.*
16. Figure 4 appears unclear to me: the upper left weight matrix in Fig 4a has 2 high entries (white pixels), but only 1 remain after binarization in Fig 4b. Also, the colormap used is rather counterintuitive: black/white would be much more suited to display binary weights.  *=> Thank you for advice for colormap.*



### Unanswered weakness

2. In the definition of a BiSE neuron, it is said that the weights and bias are enforced to be positive. In theorem 1, two expressions (eq (12) and (13)) are given: if one of these is true, then the BiSE neuron is said to be activated, and a consequence of this is that weights and bias are positive. However, both equations (12) and (13) feature a sum on negative weights (last sum of the right-hand side). This is contradictory and unclear.  *=> This must be clarified in the paper. At first, we consider every set of weights. Then we find out that if the inequality is respected, then the set of weights is positive. Thus, we enforce positivity of the weights in the BiSE definition because only the positive weights are of interest. This is a justification of the positivity of weights inside the BISE neuron.*
7. I understand the necessity of defining a dissimilarity function for the binarization of a non-activated BiSE neuron. A dissimilarity is proposed by equation (17), what is the motivation behind this particular definition? It is said to be a first choice, does it mean that other dissimilarities have been implemented and compared?  *=> This is an interesting topic to discuss. We could clarify in the paper our choice for this dissimilarity function. The first choice is to compute how far away the BiSE parameters are from the activation. The set of activated parameters is a convex set, and this is basically the distance of the bias to this convex set. As said in the future work, finding better dissimilarity function is under study.*
9. I understand that when the BiSE and LUI are activated, it means that the conducted operation can be well approximated by a binary operation. When they are not activated however, the binarization is "forced" and this is rather intriguing. What would be the cause of a layer not activated? Is it because the learning has been stopped too early? Or is it simply because there is no binary operation to be approximated here? In which case, I assume that forcing a binarization would yield some errors/inconsistencies pretty much like using a standard convolutional layer that has not yet converged? The effects of forcing a binarization on a non-activated BiSE/LUI layer should be further presented and discussed, as it is a key point of the proposed method.  *=> Thank you for your discussion. We agree that it should be more discussed. If the BiSE is activated, it can be exactly replaced by the morphological operator, it is not an approximation. Some activations do not happen because nothing enforces that it should. Sometimes we reach perfect result without needing the activation to be complete.*
15. Authors say that they have compared their results with those of LMorph and SMorph layers, but they do not show any of those comparisons  *=> We only explain the results we have on LMorph and SMorph*.
