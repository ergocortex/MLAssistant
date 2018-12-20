/*------------------------------------------------------------------------------
auth | Roberto Peribáñez Iglesias (ergocortex) 2018
------------------------------------------------------------------------------*/

#include "tree.h"

using namespace ML;

//------------------------------------------------------------------------| Node

Node::Node(void) : leaf(false) {}

Node::Node(Variant data, bool leaf) : data(data), leaf(leaf)  {}

//------------------------------------------------------------------------| Edge

Edge::Edge(const Variant &data, float p, ubyte mathop, Node *source, Node *target) :
    source(source), target(target), data(data), mathop(mathop), p(p)  {}

//------------------------------------------------------------------------| AttributeSelection

std::wstring AttributeSelection::InformationGain(DataFrame &subsamples,
    std::vector<std::wstring> &subattributes, const ubyte classes)
/*------------------------------------------------------------------------------
nots | . information gain is greater the less homogeneity an attribute has.
------------------------------------------------------------------------------*/
{
    std::wstring attribute = L"";
    std::map <std::wstring, float> informationGain;

    float entropy = subsamples.attributes.back()->GetEntropy();

    for(uint i = 0, n = subsamples.attributes.size() - classes; i < n; ++i)
    {
        auto it = std::find(subattributes.begin(), subattributes.end(), subsamples.attributes[i]->name);

        if(it != subattributes.end())
        {
            std::vector<ML::Attribute::ProbabilityDistribution> &frecuencyDetail = *subsamples.attributes[i]->GetProbabilityDistribution();

            float gain = 0.0f;
            float N = subsamples.attributes[i]->Size();

            for(uint j = 0, m = frecuencyDetail.size(); j < m; ++j)
            {
                gain -= ((frecuencyDetail[j].p / N) * subsamples.attributes.back()->GetEntropy(frecuencyDetail[j].indexes));
            }

            informationGain.insert(std::pair<std::wstring, float>(subsamples.attributes[i]->name, entropy + gain));
        }
    }

    auto it = informationGain.begin();
    int maximum = ML::FrecuencyMax<std::wstring, float>(informationGain);
    std::advance(it, maximum);

    return(it->first);
}

//------------------------------------------------------------------------| Tree

Tree::Tree(void) {}

Node *Tree::AddNode(void)
{
    Node *node = new Node();

    nodes.push_back(node);

    return(node);
}

Edge *Tree::AddEdge(const Variant &data, float p, ubyte mathop, Node *source, Node *target)
{
    Edge *edge = new Edge(data, p, mathop, source, target);

    edges.push_back(edge);

    return(edge);
}

void Tree::ClearAttribute(const std::wstring &attribute, std::vector <std::wstring> &attributes)
{
    for(uint i = 0, n = attributes.size(); i < n; ++i)
    {
        if(attributes[n - i - 1] == attribute)
            attributes.erase(attributes.begin() + (n - i - 1));
    }
}

void Tree::RankHierarchy(void)
{
    hierarchy.clear();

    if(nodes.empty()) return;

    hierarchy.insert(std::pair<Node *, Hierarchy>(nodes[0], Hierarchy(nullptr)));

    for(uint i = 0, n = edges.size(); i < n; ++i)
        hierarchy.insert(std::pair<Node *, Hierarchy>(edges[i]->target, Hierarchy(edges[i]->source)));

    for(uint i = 0, n = edges.size(); i < n; ++i)
        hierarchy.find(edges[i]->source)->second.edges.push_back(edges[i]);
}

void Tree::Prune(Node *node)
{
    auto itActual = hierarchy.find(node);

    if(itActual->second.edges.size() == 1)
    {
        Edge *edge = itActual->second.edges.front();
        Node *parent = itActual->second.parent;
        Node *child = edge->target;

        if(parent)
        {
            auto itParent = hierarchy.find(parent);

            for(uint i = 0, n = itParent->second.edges.size(); i < n; ++i)
            {
                if(itParent->second.edges[i]->target == node)
                    itParent->second.edges[i]->target = child;
            }

            edges.erase(std::find(edges.begin(), edges.end(), edge));
            nodes.erase(std::find(nodes.begin(), nodes.end(), node));

            hierarchy.erase(itActual);

            auto itChild = hierarchy.find(child);

            itChild->second.parent = parent;
        }

        Prune(child);
    }
    else
    {
        for(uint i = 0, n = itActual->second.edges.size(); i < n; ++i)
            Prune(itActual->second.edges[i]->target);
    }
}

Node *Tree::Predict(DataFrame &sample)
{
    if(nodes.empty()) return(nullptr);

    Node *node = nodes[0];

    while(true)
    {
        if(node->leaf) break;

        std::wstring attribute = node->data.ToWString();
        uint index = sample.GetColumnByAttribute(attribute);

        std::wstring sA = sample.attributes[index]->GetCell(0).ToWString();

        auto it = hierarchy.find(node);

        int found = -1;

        for(int i = 0, n = it->second.edges.size(); i < n; ++i)
        {
            std::wstring sB = it->second.edges[i]->data.ToWString();

            if(sA == sB)
            {
                found = i;
                break;
            }
        }

        if(found >= 0)
            node = it->second.edges[found]->target;
        else
            break;
    }

    return(node);
}

void Tree::GetProbabilityClusters(Node *node, std::vector <ProbabilityCluster> &probabilityCluster, float p)
{
    auto itActual = hierarchy.find(node);

    if(node->leaf)
    {
        bool found = false;

        for(uint i = 0, n = probabilityCluster.size(); i < n; ++i)
        {
            if(probabilityCluster[i].key == node->data)
            {
                probabilityCluster[i].p += p;
                probabilityCluster[i].nodes.push_back(node);
                found = true;
                break;
            }
        }

        if(!found)
        {
            probabilityCluster.push_back(ProbabilityCluster(node->data, p));
            probabilityCluster.back().nodes.push_back(node);
        }

        return;
    }

    for(uint i = 0, n = itActual->second.edges.size(); i < n; ++i)
    {
        Edge *edge = itActual->second.edges[i];
        Node *child = edge->target;

        GetProbabilityClusters(child, probabilityCluster, p * edge->p);
    }
}
