/*------------------------------------------------------------------------------
auth | Roberto Peribáñez Iglesias (ergocortex) 2018
------------------------------------------------------------------------------*/

#include <algorithm>

#include "tree.h"
#include "decision.h"


using namespace ML;

//------------------------------------------------------------------------| DecisionTree

int DecisionTree::GetArgumentIndex(const std::wstring &value, uint index)
{
    uint separator = value.find(L":");

    switch(index)
    {
    case 0 : return(stoi(value.substr(0, separator)));
    case 1 : return(stoi(value.substr(separator + 1)));
    }
}

DecisionTree::DecisionTree(ubyte attributeSelection) : Tree(), attributeSelection(attributeSelection) {}

void DecisionTree::Train(const DataFrame *dataframe)
{
    if(!dataframe) dataframe = &samples;

    DataFrame subsamples = *dataframe;

    std::vector <std::wstring> subattributes;

    for(uint i = 0, n = dataframe->attributes.size() - 1; i < n; ++i)
        subattributes.push_back(dataframe->attributes[i]->name);

    clrptrvector<Node *>(nodes);
    clrptrvector<Edge *>(edges);

    TreeInduction(subsamples, subattributes);

    RankHierarchy();
}

void DecisionTree::KCrossValidation(uint k)
{
    // '--> Reset Confusion Matrix

    confusionMatrix.Clear();

    std::vector<ML::Attribute::ProbabilityDistribution> *probabilityDistribution =
        samples.attributes.back()->GetProbabilityDistribution();

    for(uint i = 0, n = probabilityDistribution->size(); i < n; ++i)
    {
        WStringAttribute *wstringAttribute = new WStringAttribute((*probabilityDistribution)[i].value.ToWString());

        wstringAttribute->cells.insert(wstringAttribute->cells.begin(), n, L"0:0");

        confusionMatrix.attributes.push_back(wstringAttribute);
    }

    // '--> Populate Confusion Matrix

    uint delta = (samples.Size() / k) + 1;

    for(uint i = 0; i < k; ++i)
    {
        std::vector <uint> trainingIndexes;
        std::vector <uint> validationIndexes;

        uint intervalstart = (uint)((float)(i) * (float)(delta));
        uint intervalend = min(intervalstart + delta, samples.Size());

        for(uint j = 0, m = samples.Size(); j < m; ++j)
        {
            if((j >= intervalstart) && (j < intervalend))
                validationIndexes.push_back(j);
            else
                trainingIndexes.push_back(j);
        }

        DataFrame *training = samples.GetSubDataFrame(trainingIndexes);
        DataFrame *validation = samples.GetSubDataFrame(validationIndexes);

        Train(training);

        for(uint j = 0, m = validation->Size(); j < m; ++j)
        {
            int positives, instances;

            DataFrame *sample = validation->GetSubDataFrame({j});

            ML::Node *node = Predict(*sample);

            int real = GetConfusionIndex(sample->attributes.back()->GetCell(0).ToWString());

            positives = GetArgumentIndex(static_cast<WStringAttribute *>(confusionMatrix.attributes[real])->cells[real], 0);
            instances = GetArgumentIndex(static_cast<WStringAttribute *>(confusionMatrix.attributes[real])->cells[real], 1);

            ++instances;

            static_cast<WStringAttribute *>(confusionMatrix.attributes[real])->cells[real] = std::to_wstring(positives) +
                L":" + std::to_wstring(instances);

            if(node->leaf)
            {
                std::vector<Attribute *> &attributes = confusionMatrix.attributes;

                int classified = GetConfusionIndex(node->data.ToWString());

                positives = GetArgumentIndex(static_cast<WStringAttribute *>(confusionMatrix.attributes[classified])->cells[real], 0);
                instances = GetArgumentIndex(static_cast<WStringAttribute *>(confusionMatrix.attributes[classified])->cells[real], 1);

                ++positives;

                static_cast<WStringAttribute *>(confusionMatrix.attributes[classified])->cells[real] = std::to_wstring(positives) +
                    L":" + std::to_wstring(instances);
            }
        }
    }

    return;

    // '--> Mean Calculation

    for(uint i = 0, n = confusionMatrix.attributes.size(); i < n; ++i)
    {
        FloaAttribute *floatAttribute = static_cast<FloaAttribute *>(confusionMatrix.attributes[i]);

        for(uint j = 0, m = floatAttribute->Size();  j < m; ++j)
            floatAttribute->cells[j] /= (float)(k);
    }
}

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

    std::wstring attribute;

    switch(attributeSelection)
    {
    case 0 : attribute = AttributeSelection::InformationGain(subsamples, subattributes); break;
    case 1 : attribute = AttributeSelection::GiniImpurity(subsamples, subattributes); break;
    case 2 : attribute = AttributeSelection::ProportionGain(subsamples, subattributes); break;
    }

    // '--> P5 : Clear attribute selected from attribute list.

    ClearAttribute(attribute, subattributes);

    // '--> P6 : Label node with selected atributte.

    node->data = attribute;

    // '--> P7 : For every value of attribute (being subsubsample the set of elements with value V in attribute A).
    // '--> P8 : If subsubsample is empty then create an edge with mode class.
    // '--> P9 : Else create an edge than bind node to node returned frome TreeInduction(subsubdataframe, subsubattributes)

    Attribute *factor = subsamples.attributes[subsamples.GetColumnByAttribute(attribute)];

    std::vector<ML::Attribute::ProbabilityDistribution> *probabilityDistribution = factor->GetProbabilityDistribution();

    for(uint i = 0, n = (*probabilityDistribution).size(); i < n; ++i)
    {
        Node *child = nullptr;

        if((*probabilityDistribution)[i].indexes.empty())
        {
            child = AddNode();

            child->leaf = true;
            child->data = subsamples.attributes.back()->GetMode();
        }
        else
        {
            if((maxdeep == 0) || (deep < maxdeep))
            {
                child = TreeInduction(*subsamples.GetSubDataFrame((*probabilityDistribution)[i].indexes), subattributes);
            }
        }

        if(child) AddEdge((*probabilityDistribution)[i].value, (*probabilityDistribution)[i].p,
                          (*probabilityDistribution)[i].mathop, node, child);
    }

    delete(probabilityDistribution);

    --deep;
    return(node);
}

int DecisionTree::GetConfusionIndex(const std::wstring &value)
{
    for(uint i = 0, n = confusionMatrix.attributes.size(); i < n; ++i)
    {
        if(confusionMatrix.attributes[i]->name == value)
            return(i);
    }

    return(-1);

}
