\newcommand{\opecloselt}[5]{
    \begin{minipage}{.12\linewidth}
          \center{\textcolor{#3}{\checkmark}}
          \includegraphics[width=\textwidth]{#1}
        
          \center{\textcolor{#4}{\checkmark}}
          \includegraphics[width=\textwidth]{#2}
          \center{$#5$}
    \end{minipage}
}        

     \begin{tabular}{ | c | c | c | c | c | c | c | c | c | c |}
        \hline
        Dataset & \multicolumn{3}{c|}{Diskorect}  & \multicolumn{3}{c|}{MNIST} & \multicolumn{3}{c|}{Inverted MNIST} \\
        \hline
        Operation & Disk & Stick & Cross & Disk & Stick & Cross & Disk & Stick & Cross \\
        \hline
        Target
        & \begin{minipage}{.12\linewidth}
          \includegraphics[width=\textwidth]{selem_results/true_disk7.eps}
        \end{minipage}
        & \begin{minipage}{.12\linewidth}
          \includegraphics[width=\textwidth]{selem_results/true_hstick7.eps}
        \end{minipage}
        & \begin{minipage}{.12\linewidth}
          \includegraphics[width=\textwidth]{selem_results/true_dcross7.eps}
        \end{minipage}
        & \begin{minipage}{.12\linewidth}
          \includegraphics[width=\textwidth]{selem_results/true_disk7.eps}
        \end{minipage}
        & \begin{minipage}{.12\linewidth}
          \includegraphics[width=\textwidth]{selem_results/true_hstick7.eps}
        \end{minipage}
        & \begin{minipage}{.12\linewidth}
          \includegraphics[width=\textwidth]{selem_results/true_dcross7.eps}
        \end{minipage}
        & \begin{minipage}{.12\linewidth}
          \includegraphics[width=\textwidth]{selem_results/true_disk7.eps}
        \end{minipage}
        & \begin{minipage}{.12\linewidth}
          \includegraphics[width=\textwidth]{selem_results/true_hstick7.eps}
        \end{minipage}
        & \begin{minipage}{.12\linewidth}
          \includegraphics[width=\textwidth]{selem_results/true_dcross7.eps}
        \end{minipage}
        \\
        \hline
        Opening $\circ$
        & 
        \opecloselt{selem_results/diskorect_opening_disk1.eps}{selem_results/diskorect_opening_disk2.eps}{ko}{ko}{0.072}
        & 
        \opecloselt{selem_results/diskorect_opening_stick1.eps}{selem_results/diskorect_opening_stick2.eps}{ok}{ok}{0.000}
        & 
        \opecloselt{selem_results/diskorect_opening_cross1.eps}{selem_results/diskorect_opening_cross2.eps}{ok}{ok}{0.002}
        & 
        \opecloselt{selem_results/mnist_opening_disk1.eps}{selem_results/mnist_opening_disk2.eps}{ko}{ok}{0.008}
        & 
        \opecloselt{selem_results/mnist_opening_stick1.eps}{selem_results/mnist_opening_stick2.eps}{ok}{ok}{0.000}
        & 
        \opecloselt{selem_results/mnist_opening_cross1.eps}{selem_results/mnist_opening_cross2.eps}{ko}{ok}{0.001}
        & 
        \opecloselt{selem_results/inverted_mnist_opening_disk1.eps}{selem_results/inverted_mnist_opening_disk2.eps}{reredd}{}{0.006}
        & 
        \opecloselt{selem_results/inverted_mnist_opening_stick1.eps}{selem_results/inverted_mnist_opening_stick2.eps}{redblue}{}{0.001}
        & 
        \opecloselt{selem_results/inverted_mnist_opening_cross1.eps}{selem_results/inverted_mnist_opening_cross2.eps}{redred}{}{0.012}
        \\
        \hline
        Closing $\bullet$
        & 
        \opecloselt{selem_results/diskorect_closing_disk1.eps}{selem_results/diskorect_closing_disk2.eps}{ko}{ko}{0.038}
        & 
        \opecloselt{selem_results/diskorect_closing_stick1.eps}{selem_results/diskorect_closing_stick2.eps}{ok}{ok}{0.000}
        & 
        \opecloselt{selem_results/diskorect_closing_cross1.eps}{selem_results/diskorect_closing_cross2.eps}{ok}{ko}{0.000}
        & 
        \opecloselt{selem_results/mnist_closing_disk1.eps}{selem_results/mnist_closing_disk2.eps}{ko}{ko}{0.009}
        & 
        \opecloselt{selem_results/mnist_closing_stick1.eps}{selem_results/mnist_closing_stick2.eps}{ok}{ko}{0.001}
        & 
        \opecloselt{selem_results/mnist_closing_cross1.eps}{selem_results/mnist_closing_cross2.eps}{ko}{ko}{0.020}
        & 
        \opecloselt{selem_results/inverted_mnist_closing_disk1.eps}{selem_results/inverted_mnist_closing_disk2.eps}{ko}{ko}{0.009}
        & 
        \opecloselt{selem_results/inverted_mnist_closing_stick1.eps}{selem_results/inverted_mnist_closing_stick2.eps}{ok}{ok}{0.000}
        &
        \opecloselt{selem_results/inverted_mnist_closing_cross1.eps}{selem_results/inverted_mnist_closing_cross2.eps}{ok}{ok}{0.000}
        \\
        \hline
      \end{tabular}
