ASBot Chess (Abstract Strategy Bot for Chess) is a Convolutional Recurrent Neural Network (CRNN) chess engine.

Author Nicholas Entzi

The foundational ideas behind this design are:

    that the search and evaluation of chess positions involves the extraction of similar features at various "depths" of the evaluation/search
    
    in chess engines that pair an evalution function with a search algorthm, depth is far more valuable in the evaluation function
    
    the evaluation of a position should result in information pertaining to the value of all moves in the position, not just one

Inside this project:

    CRNN module design
    
    Reinforcement learning algorithm
    
    Parallelized Efficient Monte Carlo/Markov Tree Search
    
    Supervised training algorithm
    
    Simple UCI interface (Universal Chess Interface) that is Lichess compliant
    
    Other things I built while making this

tech stack standouts:

    pytorch
    
    python-chess

This is only version 1! Many improvements to come.

Huge thanks to:

    Adam Pazke for his pytorch reinforcement learning tutorial
    
    Andrew Healey for his blog on building his own chess engine
