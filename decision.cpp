/*------------------------------------------------------------------------------
auth | Roberto Peribáñez Iglesias (ergocortex) 2018
------------------------------------------------------------------------------*/

#include <algorithm>

#include "tree.h"
#include "decision.h"


using namespace ML;

//------------------------------------------------------------------------| DecisionTree

DecisionTree::DecisionTree(void) : Tree(){}

ML::Node *DecisionTree::TreeInduction(DataFrame &subsamples, std::vector<std::wstring> &subattributes)
/*------------------------------------------------------------------------------
vars | maxdeep : used for debugging.
nots | . assuming last factor is class
------------------------------------------------------------------------------*/
{
    const int maxdeep = 0;

    static int deep = 0;

    ++deep;

    // '--> P1 : Create node.

    Node *node = AddNode();

    // '--> P2 : If all the subsamples belongs to same class, then return node as leaf node of class C.
    // '--> P3 : If subattributes is empty then return node as leaf node.

    bool uniformity = subsamples.attributes.back()->GetUniformity();

    if(uniformity || subattributes.empty())
    {
        node->data = subsamples.attributes.back()->GetMode();
        node->leaf = true;

        --deep;
        return(node);
    }

    // '--> P4 : Select the attribute that best divides the subsamples dataframe.

    std::wstring attribute = AttributeSelection::InformationGain(subsamples, subattributes);

    // '--> P5 : Clear attribute selected from attribute list.

    ClearAttribute(attribute, subattributes);

    // '--> P6 : Label node with selected atributte.

    node->data = attribute;

    // '--> P7 : For every value of attribute (being subsubsample the set of elements with value V in attribute A).
    // '--> P8 : If subsubsample is empty then create an edge with mode class.
    // '--> P9 : Else create an edge than bind node to node returned frome TreeIndction(subsubdataframe, subsubattributes)

    Attribute *factor = subsamples.attributes[subsamples.GetColumnByAttribute(attribute)];

    std::vector<ML::Attribute::ProbabilityDistribution> &probabilityDistribution = *factor->GetProbabilityDistribution();

    for(uint i = 0, n = probabilityDistribution.size(); i < n; ++i)
    {
        Node *child = nullptr;

        if(probabilityDistribution[i].indexes.empty())
        {
            child = AddNode();

            child->leaf = true;
            child->data = subsamples.attributes.back()->GetMode();
        }
        else
        {
            if((maxdeep == 0) || (deep < maxdeep))
            {
                child = TreeInduction(*subsamples.GetSubDataFrame(probabilityDistribution[i].indexes), subattributes);
            }
        }

        if(child) AddEdge(probabilityDistribution[i].value, probabilityDistribution[i].p, 0, node, child);
    }

    --deep;
    return(node);
}

void DecisionTree::Build(void)
{
    DataFrame subsamples = samples;

    std::vector <std::wstring> subattributes;

    for(uint i = 0, n = samples.attributes.size() - 1; i < n; ++i)
        subattributes.push_back(samples.attributes[i]->name);

    clrptrvector<Node *>(nodes);
    clrptrvector<Edge *>(edges);

    TreeInduction(subsamples, subattributes);

    RankHierarchy();
}

void DecisionTree::KCrossValidation(uint k)
{

}
