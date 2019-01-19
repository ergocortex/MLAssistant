/*------------------------------------------------------------------------------
auth | Roberto Peribáñez Iglesias (ergocortex) 2018
------------------------------------------------------------------------------*/

#ifndef TREE_H
#define TREE_H

#include "core.h"

namespace ML
{
//------------------------------------------------------------------------| Node

class Node
/*------------------------------------------------------------------------------
vars | leaf | false : data = attribute | true : data = value (class)
------------------------------------------------------------------------------*/
{
public :

    Variant data;

    bool leaf;

public :

    Node(void);
    Node(Variant data, bool leaf);
};

//------------------------------------------------------------------------| Edge

class Edge
{
public :

    Node *source;
    Node *target;

    Variant data;

    MathOp mathop;

    float p;

public :

    Edge(const Variant &data, float p, MathOp mathop = 0, Node *source = nullptr, Node *target = nullptr);
};

//------------------------------------------------------------------------| Hierarchy

class Hierarchy
{
public :

    Node *parent;

    std::vector <Edge *> edges;

public :

    Hierarchy(Node *parent) : parent(parent) {}
};

//------------------------------------------------------------------------| AttributeSelection

class AttributeSelection
{
public :

    static std::wstring InformationGain(DataFrame &subsamples, std::vector<std::wstring> &subattributes,
        const ubyte classes = 1);

    static std::wstring GiniImpurity(DataFrame &subsamples, std::vector<std::wstring> &subattributes,
        const ubyte classes = 1);

    static std::wstring ProportionGain(DataFrame &subsamples, std::vector<std::wstring> &subattributes,
        const ubyte classes = 1);
};

//------------------------------------------------------------------------| Tree

class Tree
{
public :

    struct ProbabilityCluster
    {
    public :

        Variant key;
        float p;
        std::vector <Node *> nodes;

    public :

        ProbabilityCluster(Variant key, float p) : key(key), p(p) {}
    };

public :

    DataFrame samples;

    std::vector <Node *> nodes;
    std::vector <Edge *> edges;

    std::map<Node *, Hierarchy> hierarchy;

public :

    Tree(void);

    Node *AddNode(void);
    Edge *AddEdge(const Variant &data, float p, MathOp mathop = 0, Node *source = nullptr, Node *target = nullptr);

    void Clear(void);

    void ClearAttribute(const std::wstring &attribute, std::vector <std::wstring> &attributes);

    void RankHierarchy(void);

    void Prune(Node *node);

    Node *Predict(DataFrame &sample);

    void GetProbabilityClusters(Node *node, std::vector <ProbabilityCluster> &probabilityCluster, float p = 1.0f);
};
}

#endif // TREE_H
