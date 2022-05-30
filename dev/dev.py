\subsubsection{Reviewer 1 0DA2}

\textbf{Importance/Relevance}:Of sufficient interest\\
\textbf{Novelty/Originality/Contribution}:Moderately original\\
\textbf{Technical Correctness}:Probably correct\\
\textbf{Experimental Validation}:Lacking in some respect\\
\textbf{Clarity of Presentation}:Clear enough\\
\textbf{Reference to Prior Work}:References adequate\\

\textbf{Additional comments to author(s)}\\

The authors appear to be breaking new ground with the theory and creation of binary morphological neural nets. The theory appears sound, but the experimental results appear somewhat lacking. It is good that the MNIST dataset was one of the chosen example datasets, but no accuracy, timing or memory consumption numbers were given in comparison to traditional CNNs.

\textbf{Answer}: We thank the reviewer for their reading. We agree that the experimental results are lacking because this is a preliminary work and the experimental results are still under investigation. Further work is being done to perform extensive and rigorous experiments to analyze the practical behaviors of such networks. \\
Concerning MNIST, the model described in the paper is not adapted to classification. The training time, inference and memory consumption is the same as CNNs as the neural network is heavily based on the convolutional layer.


\subsubsection{Reviewer 2 1C23}
  

\textbf{Importance/Relevance}:Of broad interest\\
\textbf{Novelty/Originality/Contribution}:Very original\\
\textbf{Technical Correctness}:Definitely correct\\
\textbf{Experimental Validation}:Limited but convincing\\
\textbf{Clarity of Presentation}:Very clear\\
\textbf{Reference to Prior Work}:References adequate\\

\textbf{Additional comments to author(s)}\\

This paper presents a morphological neural network based on 'unfolding' the erosion and dilation operators by CNNs on binary images. In my opinion, the paper has an interesting idea when replacing the CNN convolutions with morphological operations which is theoretically explainable. However, one of the limitations of the paper is that it has no comparison with the state-of-the-art methods, par example, the papers in [4-9].

\textbf{Answer}: We thank the reviewer for their reading. We agree that comparison with the state-of-the-art methods is necessary. All methods cited in [4-9] do not solve exactly the same task. For example, [4] did not learn the morphological operator, but only the structuring element. [5-6] deal with grey-level morphology and apply a linear combination of dilation and erosion: we believe this is not perfectly adapted to binary morphology. [7] did not publish its training parameters or initialization and we did not manage to make it converge. That's why we only compared to [9], which is an improvement on [8].\\


\subsubsection{Reviewer 3 189C}
\textbf{Importance/Relevance}:Of limited interest\\
\textbf{Novelty/Originality/Contribution}:Moderately original\\
\textbf{Technical Correctness}:Probably correct\\
\textbf{Experimental Validation}:Limited but convincing\\
\textbf{Clarity of Presentation}:Difficult to read\\
\textbf{Reference to Prior Work}:References adequate\\

\textbf{Additional comments to author(s)}\\

In this work, the authors create a CNN that handles binary inputs and outputs. They replace conv with erosion and dilation.
I can understand what the authors are trying to do but I do not understand why they are trying to do it. I am not familiar with Mathematical morphology so I might not be the right person to review this paper.

\textbf{Answer}: We thank the reviewer for their reading. We motivate this work by noticing that the convolution operation is not the most adapted to deal with binary images, compared to mathematical morphology. Binary images are a type of shape representation. Our aim is to provide a framework to efficiently apply deep learning to shape analysis. \\

\subsubsection{Reviewer 4 1C8A}

\textbf{Importance/Relevance}:Of sufficient interest\\
\textbf{Novelty/Originality/Contribution}:Moderately original\\
\textbf{Technical Correctness}:Probably correct\\
\textbf{Experimental Validation}:Sufficient validation/theoretical paper\\
\textbf{Clarity of Presentation}:Very clear\\
\textbf{Reference to Prior Work}:References adequate\\

\textbf{Additional comments to author(s)}\\

The authors present learnable morphological operators to replace typical components of CNNs (conv filter, nonlinearity, pooling). They propose a Binary Structuring Element (BiSE) neuron, which can learn erosion and dilation operators using the convolution operator. They use a differentiable non-linearity and almost Binary Image representation for enabling gradient back-propagation. The results show almost perfect dice errors of 0.

I understand that the paper is geared towards theoretical development but it would be nice to see some practical applications using the learned morphological operators.

\textbf{Answer}: We thank the reviewer for their reading. We agree that a practical application is necessary, and we believe that a lot of applications could use these type of networks. However, the scope of the paper is to introduce and validate the model by showing that we can learn simple morphological operators. More sophisticated applications are under investigation for further publications.\\
