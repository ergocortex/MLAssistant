/*------------------------------------------------------------------------------
auth | Roberto Peribáñez Iglesias (ergocortex) on Dec. 2018
------------------------------------------------------------------------------*/

#ifndef CORE_H
#define CORE_H

#include <map>
#include <vector>
#include <string>
#include <algorithm>

typedef unsigned char ubyte;
typedef unsigned int  uint;

#define safedelete(ptr); {delete(ptr); ptr = nullptr;}
#define FLTEPSILON 0.000001f


namespace ML
{
//------------------------------------------------------------------------| Global

inline bool fcmp(const float &a, const float &b, const float epsilon = FLTEPSILON)
{
    return(fabs(b - a) < epsilon);
}

inline float max(const float cmpA, const float cmpB)
{
    return((cmpA > cmpB) ? cmpA : cmpB);
}

inline float min(const float cmpA, const float cmpB)
{
    return((cmpA < cmpB) ? cmpA : cmpB);
}

template <class T> void clrptrvector(std::vector <T> &ref)
{
    while(!ref.empty())
    {
        safedelete(ref.back());
        ref.pop_back();
    }
}

template <class K> void FrecuencyMapping(std::map <K, int> &values, const K &value)
{
    auto it = values.find(value);

    if(it == values.end())
        values.insert(values.end(), std::pair <K, int> (value, 1));
    else
        ++it->second;
}

template <class K, class V> int FrecuencyMax(std::map <K, V> &values)
{
    int index = -1;
    V maximum = -std::numeric_limits<V>::max();

    int i = 0;

    for(auto it = values.begin(); it != values.end(); ++it, ++i)
    {
        if(it->second > maximum)
        {
            index = i;
            maximum = it->second;
        }
    }

    return(index);
}

template <class K, class V> int FrecuencyMin(std::map <K, V> &values)
{
    int index = -1;
    V minimum = +std::numeric_limits<V>::max();

    int i = 0;

    for(auto it = values.begin(); it != values.end(); ++it, ++i)
    {
        if(it->second < minimum)
        {
            index = i;
            minimum = it->second;
        }
    }

    return(index);
}

template <class K> K FrecuencyMode(std::map <K, int> &values)
{
    auto it = values.begin();

    std::advance(it, FrecuencyMax<K, int>(values));

    return(it->first);
}

template <class K> typename std::map<K, int>::iterator FrecuencyMedian(std::map<K, int> &values, uint N, uint &f)
{
    auto it = values.begin();

    for(f = 0; f < (N / 2); ++it, f += it->second);

    return(it);
}

/*------------------------------------------------------------------------| MathOp
vars | mathop | 0 : = | 1 : < | 2 : <= | 3 : >= | 4 : >
------------------------------------------------------------------------------*/

typedef ubyte MathOp;

//------------------------------------------------------------------------| Variant

struct Variant
/*------------------------------------------------------------------------------
desc | . simulates dynamic typing
------------------------------------------------------------------------------*/
{
public :

    enum Type {Generic, Bool, Int, Float, WString};

    union Data
    {
        bool b;
        int i;
        float f;
        void *ptr;
    };

public :

    Type type;
    Data data;

public :

    Variant(void);
    Variant(bool b);
    Variant(int i);
    Variant(float f);
    Variant(const std::wstring &wstring);

    bool operator==(const Variant &rhs) const;
    bool operator<(const Variant &rhs) const;
    bool operator<=(const Variant &rhs) const;
    bool operator>=(const Variant &rhs) const;
    bool operator>(const Variant &rhs) const;

    bool IsNull(void);

    std::wstring ToWString(void) const;

private :

    std::wstring GetWString(void) const;
};

//------------------------------------------------------------------------| Attribute

struct Attribute
/*------------------------------------------------------------------------------
nots | . entropy in samples with more than 2 classes can be greater than 1.
------------------------------------------------------------------------------*/
{
public :

    struct ProbabilityDistribution
    {
    public :

        Variant value;
        float p;
        MathOp mathop;

        std::vector <uint> indexes;

    public :

        ProbabilityDistribution(Variant value, float p, MathOp mathop = 0) : value(value), p(p), mathop(mathop) {}
    };

public :

    std::wstring name;
    bool discrete;

public :

    Attribute(const std::wstring &name, const bool discrete);

protected :

    template <class T> std::map<T, int> GetFrecuencyMapping(const std::vector <T> &cells,
        const std::vector<uint> &restrictions = {})
    /*--------------------------------------------------------------------------
    vars | indexes : index mapping restriction
    --------------------------------------------------------------------------*/
    {
        std::map<T, int> frecuency;

        if(restrictions.empty())
        {
            for(uint i = 0, n = cells.size(); i < n; ++i)
                FrecuencyMapping<T>(frecuency, cells[i]);
        }
        else
        {
            for(uint i = 0, n = cells.size(); i < n; ++i)
            {
                if(std::find(restrictions.begin(), restrictions.end(), i) != restrictions.end())
                    FrecuencyMapping<T>(frecuency, cells[i]);
            }
        }

        return(frecuency);
    }

    template <class T> std::vector<ProbabilityDistribution> *GetDistributionFuncion(
        const std::vector <T> &cells, const std::vector<uint> &restrictions = {})
    /*--------------------------------------------------------------------------
    nots | . for discrete variables, based on distribution function.
    --------------------------------------------------------------------------*/
    {
        std::vector <ProbabilityDistribution> *probabilityDistribution = new std::vector <ProbabilityDistribution>;

        std::map<T, int> frecuency = GetFrecuencyMapping<T>(cells, restrictions);

        float N = cells.size();

        for(auto it : frecuency)
        {
            probabilityDistribution->push_back(ProbabilityDistribution(it.first, (float)(it.second)/N));

            if(restrictions.empty())
            {
                for(uint i = 0, n = cells.size(); i < n; ++i)
                {
                    if(cells[i] == it.first)
                        probabilityDistribution->back().indexes.push_back(i);
                }
            }
            else
            {
                for(uint i = 0, n = cells.size(); i < n; ++i)
                {
                    if(std::find(restrictions.begin(), restrictions.end(), i) != restrictions.end())
                    {
                        if(cells[i] == it.first)
                            probabilityDistribution->back().indexes.push_back(i);
                    }
                }
            }
        }

        return(probabilityDistribution);
    }

    template <class T> std::vector<ProbabilityDistribution> *GetDensityFunction(
        const std::vector <T> &cells, const std::vector<uint> &restrictions = {})
    /*--------------------------------------------------------------------------
    nots | . for continuous variables, based on density function.
    --------------------------------------------------------------------------*/
    {
        std::vector <ProbabilityDistribution> *probabilityDistribution = new std::vector <ProbabilityDistribution>;

        std::map<T, int> frecuency = GetFrecuencyMapping<T>(cells, restrictions);

        uint N = cells.size();

        typename std::map<T, int>::iterator it;

        if(N > 1)
        {
            uint p, q;

            it = FrecuencyMedian<T>(frecuency, N, p);

            q = N - p;

            probabilityDistribution->push_back(ProbabilityDistribution(it->first, (float)(p)/(float)(N), 1));
            probabilityDistribution->push_back(ProbabilityDistribution(it->first, (float)(q)/(float)(N), 3));
        }
        else
        {
            it = frecuency.begin();

            probabilityDistribution->push_back(ProbabilityDistribution(it->first, 1.0f, 0));
        }

        if(restrictions.empty())
        {
            for(uint i = 0, n = cells.size(); i < n; ++i)
            {
                if(cells[i] < it->first)
                    probabilityDistribution->front().indexes.push_back(i);
                else
                    probabilityDistribution->back().indexes.push_back(i);
            }
        }
        else
        {
            for(uint i = 0, n = cells.size(); i < n; ++i)
            {
                if(std::find(restrictions.begin(), restrictions.end(), i) != restrictions.end())
                {
                    if(cells[i] < it->first)
                        probabilityDistribution->front().indexes.push_back(i);
                    else
                        probabilityDistribution->back().indexes.push_back(i);
                }
            }
        }

        return(probabilityDistribution);
    }

    template <class T> float GetEntropy(const std::vector <T> &cells,
        const std::vector <uint> &restrictions)
    {
        std::map<T, int> frecuency = GetFrecuencyMapping<T>(cells, restrictions);

        std::vector<float> proportion;

        float entropy = 0.0f;
        float N = cells.size();

        for(auto it : frecuency)
            proportion.push_back((float)(it.second)/N);

        for(float p : proportion)
            entropy -= (p * log2(p));

        return(entropy);
    }

    template <class T> float GetGiniIndex(const std::vector <T> &cells,
        const std::vector <uint> &restrictions)
    {
        std::map<T, int> frecuency = GetFrecuencyMapping<T>(cells, restrictions);

        std::vector<float> proportion;

        float gini = 1.0f;
        float N = cells.size();

        for(auto it : frecuency)
            proportion.push_back((float)(it.second)/N);

        for(float p : proportion)
            gini -= (p * p);

        return(gini);
    }

public :

    virtual uint Size(void);

    virtual bool GetUniformity(void);
    virtual Variant GetMode(const std::vector<uint> &indexes = {});

    virtual float GetAttributeEntropy(const std::vector<uint> &restrictions = {});
    virtual float GetAttributeGiniIndex(const std::vector<uint> &restrictions = {});

    virtual std::vector<ProbabilityDistribution> *GetProbabilityDistribution(
        const std::vector<uint> &restriction = {});

    virtual Variant GetCell(uint index);
};

//------------------------------------------------------------------------| BoolVector

struct BoolAttribute: public Attribute
{
public :

    std::vector <bool> cells;

public :

    BoolAttribute(const std::wstring &name);

public :

    virtual uint Size(void);

    virtual bool GetUniformity(void);
    virtual Variant GetMode(const std::vector<uint> &indexes = {});

    virtual float GetAttributeEntropy(const std::vector<uint> &restrictions = {});
    virtual float GetAttributeGiniIndex(const std::vector<uint> &restrictions = {});

    virtual std::vector<ProbabilityDistribution> *GetProbabilityDistribution(
        const std::vector<uint> &restriction = {});

    virtual Variant GetCell(uint index);
};

//------------------------------------------------------------------------| IntAttribute

struct IntAttribute: public Attribute
{
public :

    std::vector <int> cells;

public :

    IntAttribute(const std::wstring &name);

public :

    virtual uint Size(void);

    virtual bool GetUniformity(void);
    virtual Variant GetMode(const std::vector<uint> &indexes = {});

    virtual float GetAttributeEntropy(const std::vector<uint> &restrictions = {});
    virtual float GetAttributeGiniIndex(const std::vector<uint> &restrictions = {});

    virtual std::vector<ProbabilityDistribution> *GetProbabilityDistribution(
        const std::vector<uint> &restriction = {});

    virtual Variant GetCell(uint index);
};

//------------------------------------------------------------------------| FloaAttribute

struct FloaAttribute: public Attribute
{
public :

    std::vector <float> cells;

public :

    FloaAttribute(const std::wstring &name);

public :

    virtual uint Size(void);

    virtual bool GetUniformity(void);
    virtual Variant GetMode(const std::vector<uint> &indexes = {});

    virtual float GetAttributeEntropy(const std::vector<uint> &restrictions = {});
    virtual float GetAttributeGiniIndex(const std::vector<uint> &restrictions = {});

    virtual std::vector<ProbabilityDistribution> *GetProbabilityDistribution(
        const std::vector<uint> &restriction = {});

    virtual Variant GetCell(uint index);
};

//------------------------------------------------------------------------| WStringAttribute

struct WStringAttribute: public Attribute
{
public :

    std::vector <std::wstring> cells;

public :

    WStringAttribute(const std::wstring &name);

public :

    virtual uint Size(void);

    virtual bool GetUniformity(void);
    virtual Variant GetMode(const std::vector<uint> &indexes = {});

    virtual float GetAttributeEntropy(const std::vector<uint> &restrictions = {});
    virtual float GetAttributeGiniIndex(const std::vector<uint> &restrictions = {});

    virtual std::vector<ProbabilityDistribution> *GetProbabilityDistribution(const std::vector<uint> &restriction = {});

    virtual Variant GetCell(uint index);
};

//------------------------------------------------------------------------| DataFrame

struct DataFrame
/*------------------------------------------------------------------------------
nots | . vector requires pointer of Vectors to avoid object slicing
------------------------------------------------------------------------------*/
{
public :

    enum AttributeType {GenericType, BoolType, IntType, FloatType, WStringType};

public :

    std::vector <Attribute *> attributes;

public :

    DataFrame(void);

    uint Size(void);
    void Clear(void);

    ubyte GetColumnByAttribute(const std::wstring &attribute);
    ubyte GetColumnType(ubyte index);

    DataFrame *GetSubDataFrame(const std::vector <uint> &indexes);
};

//------------------------------------------------------------------------| Common

bool Validate(Variant &a, MathOp &mathop, Variant &b);
}

#endif // CORE_H
