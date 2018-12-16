/*------------------------------------------------------------------------------
auth | Roberto Peribáñez Iglesias (ergocortex) 2018
------------------------------------------------------------------------------*/

#include "alpha/core/codelib.h"

#include "tree.h"
#include "probability.h"

using namespace ML;

//------------------------------------------------------------------------| ProbabilityTree

ProbabilityTree::ProbabilityTree(void) : Tree() {}

ML::Node *ProbabilityTree::TreeInduction(DataFrame &subsamples, std::vector <std::wstring> subattributes)
{
    Node *node = AddNode();

    bool leaf = (subattributes.size() == 1);

    std::wstring attribute = AttributeSelection::InformationGain(subsamples, subattributes, leaf ? 0 : 1);

    ClearAttribute(attribute, subattributes);

    node->data = attribute;

    for(Factor *factor : subsamples.factors)
    {
        if(factor->attribute == attribute)
        {
            if(factor->discrete)
            {
                std::vector<ML::Factor::FrecuencyDiscrete> &frecuencyDiscrete = *factor->GetFrecuencyDiscrete();

                for(uint i = 0, n = frecuencyDiscrete.size(); i < n; ++i)
                {
                    Node *child = nullptr;

                    if(leaf)
                    {
                        child = AddNode();

                        child->leaf = true;
                        child->data = frecuencyDiscrete[i].key;
                    }
                    else
                    {
                        child = TreeInduction(*subsamples.GetSubDataFrame(frecuencyDiscrete[i].indexes), subattributes);
                    }

                    if(child) AddEdge(frecuencyDiscrete[i].key, frecuencyDiscrete[i].p, 0, node, child);
                }
            }
            else
            {
                std::vector<ML::Factor::FrecuencyContinuous> &frecuencyContinuous = *factor->GetFrecuencyContinuous();

                for(uint i = 0, n = frecuencyContinuous.size(); i < n; ++i)
                {
                    Node *child = nullptr;

                    if(leaf)
                    {
                        child = AddNode();

                        child->leaf = true;
                        child->data = frecuencyContinuous[i].key;
                    }
                    else
                    {
                        child = TreeInduction(*subsamples.GetSubDataFrame(frecuencyContinuous[i].indexes), subattributes);
                    }

                    if(child) AddEdge(frecuencyContinuous[i].key, frecuencyContinuous[i].p, frecuencyContinuous[i].mathop, node, child);
                }
            }
        }
    }

    return(node);
}

void ProbabilityTree::Build(void)
{
    DataFrame subsamples = samples;

    std::vector <std::wstring> subattributes;

    for(uint i = 0, n = samples.factors.size(); i < n; ++i)
        subattributes.push_back(samples.factors[i]->attribute);

    phy::clrptrvector<Node *>(nodes);
    phy::clrptrvector<Edge *>(edges);

    TreeInduction(subsamples, subattributes);

    RankHierarchy();
}
