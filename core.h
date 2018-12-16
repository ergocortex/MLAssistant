/*------------------------------------------------------------------------------
auth | Roberto Peribáñez Iglesias (ergocortex) 2018
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

namespace ML
{
//------------------------------------------------------------------------| Common

template <class T> void clrptrvector(std::vector <T> &ref)
{
    while(!ref.empty())
    {
        safedelete(ref.back());
        ref.pop_back();
    }
}

template <class T> void FrecuencyMapping(std::map <T, int> &values, const T &value)
{
    auto it = values.find(value);

    if(it == values.end())
        values.insert(values.end(), std::pair <T, int> (value, 1));
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

template <class T> T FrecuencyMode(std::map <T, int> &values)
{
    auto it = values.begin();
    std::advance(it, FrecuencyMax<T, int>(values));

    return(it->first);
}

template <class T> typename std::map<T, int>::iterator FrecuencyMedian(std::map<T, int> &values, uint N, uint &p)
{
    auto it = values.begin();

    for(p = 0; p < (N / 2); ++it, p += it->second);

    return(it);
}

//------------------------------------------------------------------------| Variant

struct Variant
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

    bool IsNull(void);

    std::wstring ToWString(void) const;
};

//------------------------------------------------------------------------| Factor

struct Factor
/*------------------------------------------------------------------------------
nots | . entropy in samples with more than 2 classes can be greater than 1.
------------------------------------------------------------------------------*/
{
public :

    struct FrecuencyDiscrete
    {
    public :

        Variant key;
        float p;
        std::vector <uint> indexes;

    public :

        FrecuencyDiscrete(Variant key, float p) : key(key), p(p) {}
    };

    struct FrecuencyContinuous : public FrecuencyDiscrete
    /*--------------------------------------------------------------------------
    vars | mathop | 0 : = | 1 : < | 2 : <= | 3 : >= | 4 : >
    --------------------------------------------------------------------------*/
    {
    public :

        ubyte mathop;

    public :

        FrecuencyContinuous(Variant key, ubyte mathop, float p) :
            FrecuencyDiscrete(key, p), mathop(mathop) {}
    };

public :

    std::wstring attribute;
    bool discrete;

public :

    Factor(const std::wstring &attribute, const bool discrete);

protected :

    template <class T> std::map<T, int> GetFrecuencyMapping(const std::vector <T> &cells, const std::vector<uint> &indexes = {})
    {
        std::map<T, int> frecuency;

        if(indexes.empty())
        {
            for(uint i = 0, n = cells.size(); i < n; ++i)
                FrecuencyMapping<T>(frecuency, cells[i]);
        }
        else
        {
            for(uint i = 0, n = cells.size(); i < n; ++i)
            {
                if(std::find(indexes.begin(), indexes.end(), i) != indexes.end())
                    FrecuencyMapping<T>(frecuency, cells[i]);
            }
        }

        return(frecuency);
    }

    template <class T> std::vector<FrecuencyDiscrete> *GetDiscreteFrecuency(const std::vector <T> &cells)
    {
        std::vector <FrecuencyDiscrete> *frecuencyDiscrete = new std::vector <FrecuencyDiscrete>;

        std::map<T, int> frecuency = GetFrecuencyMapping<T>(cells);

        float N = cells.size();

        for(auto it : frecuency)
        {
            frecuencyDiscrete->push_back(FrecuencyDiscrete(it.first, (float)(it.second)/N));

            for(uint i = 0, n = cells.size();i < n; ++i)
            {
                if(cells[i] == it.first)
                    frecuencyDiscrete->back().indexes.push_back(i);
            }
        }

        return(frecuencyDiscrete);
    }

    template <class T> std::vector<Factor::FrecuencyContinuous> *GetContinuousFrecuency(const std::vector <T> &cells)
    {
        std::vector <FrecuencyContinuous> *frecuencyContinuous = new std::vector <FrecuencyContinuous>;

        std::map<T, int> frecuency = GetFrecuencyMapping<T>(cells);

        uint N = cells.size();

        typename std::map<T, int>::iterator it;

        if(N > 1)
        {
            uint p, q;

            it = FrecuencyMedian<T>(frecuency, N, p);

            q = N - p;

            frecuencyContinuous->push_back(FrecuencyContinuous(it->first, 1, (float)(p)/(float)(N)));
            frecuencyContinuous->push_back(FrecuencyContinuous(it->first, 3, (float)(q)/(float)(N)));
        }
        else
        {
            it = frecuency.begin();

            frecuencyContinuous->push_back(FrecuencyContinuous(it->first, 0, 1.0f));
        }

        for(uint i = 0, n = cells.size(); i < n; ++i)
        {
            if(cells[i] < it->first)
                frecuencyContinuous->front().indexes.push_back(i);
            else
                frecuencyContinuous->back().indexes.push_back(i);
        }

        return(frecuencyContinuous);
    }

public :

    virtual uint Size(void);

    virtual bool GetUniformity(void);
    virtual Variant GetMode(const std::vector<uint> &indexes = {});

    virtual float GetEntropy(const std::vector<uint> &indexes = {});

    virtual std::vector<FrecuencyDiscrete> *GetFrecuencyDiscrete(void);
    virtual std::vector<FrecuencyContinuous> *GetFrecuencyContinuous(void);

    virtual Variant GetCell(uint index);
};

//------------------------------------------------------------------------| BoolFactor

struct BoolFactor: public Factor
{
public :

    std::vector <bool> cells;

public :

    BoolFactor(const std::wstring &attribute);

public :

    virtual uint Size(void);

    virtual bool GetUniformity(void);
    virtual Variant GetMode(const std::vector<uint> &indexes = {});

    virtual float GetEntropy(const std::vector<uint> &indexes = {});

    virtual std::vector<FrecuencyDiscrete> *GetFrecuencyDiscrete(void);

    virtual Variant GetCell(uint index);
};

//------------------------------------------------------------------------| IntFactor

struct IntFactor: public Factor
{
public :

    std::vector <int> cells;

public :

    IntFactor(const std::wstring &attribute);

public :

    virtual uint Size(void);

    virtual bool GetUniformity(void);
    virtual Variant GetMode(const std::vector<uint> &indexes = {});

    virtual float GetEntropy(const std::vector<uint> &indexes = {});

    virtual std::vector<FrecuencyDiscrete> *GetFrecuencyDiscrete(void);
    virtual std::vector<FrecuencyContinuous> *GetFrecuencyContinuous(void);

    virtual Variant GetCell(uint index);
};

//------------------------------------------------------------------------| FloatFactor

struct FloatFactor: public Factor
{
public :

    std::vector <float> cells;

public :

    FloatFactor(const std::wstring &attribute);

public :

    virtual uint Size(void);

    virtual bool GetUniformity(void);
    virtual Variant GetMode(const std::vector<uint> &indexes = {});

    virtual float GetEntropy(const std::vector<uint> &indexes = {});

    virtual std::vector<FrecuencyDiscrete> *GetFrecuencyDiscrete(void);
    virtual std::vector<FrecuencyContinuous> *GetFrecuencyContinuous(void);

    virtual Variant GetCell(uint index);
};

//------------------------------------------------------------------------| WStringFactor

struct WStringFactor: public Factor
{
public :

    std::vector <std::wstring> cells;

public :

    WStringFactor(const std::wstring &attribute);

public :

    virtual uint Size(void);

    virtual bool GetUniformity(void);
    virtual Variant GetMode(const std::vector<uint> &indexes = {});

    virtual float GetEntropy(const std::vector<uint> &indexes = {});

    virtual std::vector<FrecuencyDiscrete> *GetFrecuencyDiscrete(void);

    virtual Variant GetCell(uint index);
};

//------------------------------------------------------------------------| DataFrame

struct DataFrame
/*------------------------------------------------------------------------------
nots | . vector requires pointer of factors to avoid object slicing
------------------------------------------------------------------------------*/
{
public :

    enum FactorType {GenericType, BoolType, IntType, FloatType, WStringType};

public :

    std::vector <Factor *> factors;

public :

    DataFrame(void);

    ubyte GetColumnByAttribute(const std::wstring &attribute);
    ubyte GetColumnType(ubyte index);

    DataFrame *GetSubDataFrame(const std::vector <uint> &indexes);
};
}

#endif // CORE_H
