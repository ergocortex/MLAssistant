/*------------------------------------------------------------------------------
auth | Roberto Peribáñez Iglesias
date | Dec. 2018
------------------------------------------------------------------------------*/

#ifndef PROBABILITY_H
#define PROBABILITY_H

#include "tree.h"


namespace ML
{
//------------------------------------------------------------------------| ProbabilityTree

class ProbabilityTree : public Tree
/*------------------------------------------------------------------------------
vars | attributeSelection | 0 : Information Gain | 1 : Gini Impurity | 2 : Proportion Gain
------------------------------------------------------------------------------*/
{
public :

    ubyte attributeSelection;

public :

    ProbabilityTree(ubyte attributeSelection = 0);

    Node *TreeInduction(DataFrame &subsamples, std::vector<std::wstring> subattributes);

    void Build(void);
};
}

#endif // PROBABILITY_H
