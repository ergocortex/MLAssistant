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
nots | . assuming last factor is class
------------------------------------------------------------------------------*/
{
    const int maxdeep = 0;

    static int deep = 0;

    ++deep;

    // '--> P1 : Create node.

    Node *node = AddNode();

    // '--> P2 : If all the subsamples belongs to same class, then return node as leaf node of class C

    // '--> P3 : If subattributes is empty then then return node as leaf node of mode class.

    bool uniformity = subsamples.factors.back()->GetUniformity();

    if(uniformity || subattributes.empty())
    {
        node->data = subsamples.factors.back()->GetMode();
        node->leaf = true;

        --deep;
        return(node);
    }

    // '--> P4 : SelectSINO  aplicar  Método_Selección_Atributos(E,  Lista_Atributos)
    //           para  seleccionar el atributo A que mejor particiona E.

    std::wstring attribute = AttributeSelection::InformationGain(subsamples, subattributes);

    // '--> P5 : Borrar Atributo A de la lista de Atributos Lista_Atributos

    ClearAttribute(attribute, subattributes);

    // '--> P6 : Etiquetar N con el atributo seleccionado

    node->data = attribute;

    // '--> P7 : Para cada valor V de A
    //           Siendo Ev el subconjunto de elementos en E con valor V en el atributo A

    // '--> P8 : Si Ev esta vacio
    //           Entonces unir al nodo N una hoja etiquetada con la clase mayoritaria de E

    // '--> P9 : Sino unir al nodo N el nodo retornado de Inducir_arbol (Ev, ListaAtributos,
    //           Metodo_Seleccion_Atributos)


    Factor *factor = subsamples.factors[subsamples.GetColumnByAttribute(attribute)];

    if(factor->discrete)
    {
        std::vector<ML::Factor::FrecuencyDiscrete> &frecuencyDiscrete = *factor->GetFrecuencyDiscrete();

        for(uint i = 0, n = frecuencyDiscrete.size(); i < n; ++i)
        {
            Node *child = nullptr;

            if(frecuencyDiscrete[i].indexes.empty())
            {
                child = AddNode();

                child->leaf = true;
                child->data = subsamples.factors.back()->GetMode();
            }
            else
            {
                if((maxdeep == 0) || (deep < maxdeep))
                {
                    child = TreeInduction(*subsamples.GetSubDataFrame(frecuencyDiscrete[i].indexes), subattributes);
                }
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

            if(frecuencyContinuous[i].indexes.empty())
            {
                child = AddNode();

                child->leaf = true;
                child->data = subsamples.factors.back()->GetMode();
            }
            else
            {
                if(deep < maxdeep)
                {
                    child = TreeInduction(*subsamples.GetSubDataFrame(frecuencyContinuous[i].indexes), subattributes);
                }
            }

            if(child) AddEdge(frecuencyContinuous[i].key, frecuencyContinuous[i].p, frecuencyContinuous[i].mathop, node, child);
        }
    }

    --deep;
    return(node);
}

void DecisionTree::Train(void)
{
    DataFrame subsamples = samples;

    std::vector <std::wstring> subattributes;

    for(uint i = 0, n = samples.factors.size() - 1; i < n; ++i)
        subattributes.push_back(samples.factors[i]->attribute);

    clrptrvector<Node *>(nodes);
    clrptrvector<Edge *>(edges);

    TreeInduction(subsamples, subattributes);

    RankHierarchy();
}
