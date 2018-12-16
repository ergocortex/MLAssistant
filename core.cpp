/*------------------------------------------------------------------------------
auth | Roberto Peribáñez Iglesias (ergocortex) 2018
------------------------------------------------------------------------------*/

#include <map>

#include "core.h"

using namespace ML;

//------------------------------------------------------------------------| Variant

Variant::Variant(void) : type(Generic) {}

Variant::Variant(bool b) : type(Bool)
{
    data.b = b;
}

Variant::Variant(int i) : type(Int)
{
    data.i = i;
}

Variant::Variant(float f) : type(Float)
{
    data.f = f;
}

Variant::Variant(const std::wstring &wstring) : type(WString)
{
    std::wstring *string = new std::wstring(wstring);

    data.ptr = string;
}

bool Variant::operator==(const Variant &rhs) const
{
    return(this->ToWString() == rhs.ToWString());
}

bool Variant::IsNull(void)
{
    switch(type)
    {
    case Bool : return(false);
    case Int : return(data.i == 0);
    case Float : return(data.f == 0.0f);
    case WString :
    {
        std::wstring *wstring = static_cast<std::wstring *>(data.ptr);

        return(wstring->empty());
    }
    case Generic : return(true);
    }

    return(false);
}

std::wstring Variant::ToWString(void) const
{
    switch(type)
    {
    case Bool : return(data.b ? L"Verdadero" : L"Falso");
    case Int : return(std::to_wstring(data.i));
    case Float : return(std::to_wstring(data.f));
    case WString :
    {
        std::wstring *wstring = static_cast<std::wstring *>(data.ptr);

        return(*wstring);
    }
    case Generic : return(L"");
    }

    return(L"");
}

//------------------------------------------------------------------------| Factor

Factor::Factor(const std::wstring &attribute, const bool discrete) :
    attribute(attribute), discrete(discrete) {}

uint Factor::Size(void) {return(0);}

bool Factor::GetUniformity(void) {return(true);}

Variant Factor::GetMode(const std::vector<uint> &indexes)
{
    Variant variant(L"");

    return(variant);
}

float Factor::GetEntropy(const std::vector<uint> &/*indexes*/) {return(0.0f);}

std::vector<Factor::FrecuencyDiscrete> *Factor::GetFrecuencyDiscrete(void) {return(nullptr);}

std::vector<Factor::FrecuencyContinuous> *Factor::GetFrecuencyContinuous(void) {return(nullptr);}

Variant Factor::GetCell(uint index)
{
    Variant variant(L"");

    return(variant);
}

//------------------------------------------------------------------------| BoolFactor

BoolFactor::BoolFactor(const std::wstring &attribute) : Factor(attribute, true) {}

uint BoolFactor::Size(void) {return(cells.size());}

bool BoolFactor::GetUniformity(void)
{
    for(uint i = 1, n = cells.size(); i < n; ++i)
    {
        if(cells[i] != cells.front())
            return(false);
    }

    return(true);
}

Variant BoolFactor::GetMode(const std::vector<uint> &indexes)
{
    std::map<bool, int> frecuency = GetFrecuencyMapping<bool>(cells, indexes);

    Variant variant(FrecuencyMode<bool>(frecuency));

    return(variant);
}

float BoolFactor::GetEntropy(const std::vector<uint> &/*indexes*/)
{
    std::map<bool, int> frecuency = GetFrecuencyMapping<bool>(cells);
    std::vector<float> proportion;

    float entropy = 0.0f;
    float N = cells.size();

    for(auto it : frecuency)
        proportion.push_back((float)(it.second)/N);

    for(float p : proportion)
        entropy -= (p * log2(p));

    return(entropy);
}

std::vector<Factor::FrecuencyDiscrete> *BoolFactor::GetFrecuencyDiscrete(void)
{
    return(GetDiscreteFrecuency<bool>(cells));
}

Variant BoolFactor::GetCell(uint index)
{
    Variant variant(cells[index]);

    return(variant);
}

//------------------------------------------------------------------------| IntFactor

IntFactor::IntFactor(const std::wstring &attribute) : Factor(attribute, false) {}

uint IntFactor::Size(void) {return(cells.size());}

bool IntFactor::GetUniformity(void)
{
    for(uint i = 1, n = cells.size(); i < n; ++i)
    {
        if(cells[i] != cells.front())
            return(false);
    }

    return(true);
}

Variant IntFactor::GetMode(const std::vector<uint> &indexes)
{
    std::map<int, int> frecuency = GetFrecuencyMapping<int>(cells, indexes);

    Variant variant(FrecuencyMode<int>(frecuency));

    return(variant);
}

float IntFactor::GetEntropy(const std::vector<uint> &/*indexes*/)
{
    std::map<int, int> frecuency = GetFrecuencyMapping<int>(cells);
    std::vector<float> proportion;

    float entropy = 0.0f;
    float N = cells.size();

    for(auto it : frecuency)
        proportion.push_back((float)(it.second)/N);

    for(float p : proportion)
        entropy -= (p * log2(p));

    return(entropy);
}

std::vector<Factor::FrecuencyDiscrete> *IntFactor::GetFrecuencyDiscrete(void)
{
    return(GetDiscreteFrecuency<int>(cells));
}

std::vector<Factor::FrecuencyContinuous> *IntFactor::GetFrecuencyContinuous(void)
{
    return(GetContinuousFrecuency<int>(cells));
}

Variant IntFactor::GetCell(uint index)
{
    Variant variant(cells[index]);

    return(variant);
}

//------------------------------------------------------------------------| FloatFactor

FloatFactor::FloatFactor(const std::wstring &attribute) : Factor(attribute, false) {}

uint FloatFactor::Size(void) {return(cells.size());}

bool FloatFactor::GetUniformity(void)
{
    for(uint i = 1, n = cells.size(); i < n; ++i)
    {
        if(cells[i] != cells.front())
            return(false);
    }

    return(true);
}

Variant FloatFactor::GetMode(const std::vector<uint> &indexes)
{
    std::map<float, int> frecuency = GetFrecuencyMapping<float>(cells, indexes);

    Variant variant(FrecuencyMode<float>(frecuency));

    return(variant);
}

float FloatFactor::GetEntropy(const std::vector<uint> &/*indexes*/)
{
    std::map<float, int> frecuency = GetFrecuencyMapping<float>(cells);
    std::vector<float> proportion;

    float entropy = 0.0f;
    float N = cells.size();

    for(auto it : frecuency)
        proportion.push_back((float)(it.second)/N);

    for(float p : proportion)
        entropy -= (p * log2(p));

    return(entropy);
}

std::vector<Factor::FrecuencyDiscrete> *FloatFactor::GetFrecuencyDiscrete(void)
{
    return(GetDiscreteFrecuency<float>(cells));
}

std::vector<Factor::FrecuencyContinuous> *FloatFactor::GetFrecuencyContinuous(void)
{
    return(GetContinuousFrecuency<float>(cells));
}

Variant FloatFactor::GetCell(uint index)
{
    Variant variant(cells[index]);

    return(variant);
}

//------------------------------------------------------------------------| WStringFactor

WStringFactor::WStringFactor(const std::wstring &attribute) : Factor(attribute, true) {}

uint WStringFactor::Size(void) {return(cells.size());}

bool WStringFactor::GetUniformity(void)
{
    for(uint i = 1, n = cells.size(); i < n; ++i)
    {
        if(cells[i] != cells.front())
            return(false);
    }

    return(true);
}

Variant WStringFactor::GetMode(const std::vector<uint> &indexes)
{
    std::map<std::wstring, int> frecuency = GetFrecuencyMapping<std::wstring>(cells, indexes);

    Variant variant(FrecuencyMode<std::wstring>(frecuency));

    return(variant);
}

float WStringFactor::GetEntropy(const std::vector<uint> &indexes)
{
    std::map<std::wstring, int> frecuency = GetFrecuencyMapping<std::wstring>(cells, indexes);
    std::vector<float> proportion;

    float entropy = 0.0f;
    float N = cells.size();

    for(auto it : frecuency)
        proportion.push_back((float)(it.second)/N);

    for(float p : proportion)
        entropy -= (p * log2(p));

    return(entropy);
}

std::vector<Factor::FrecuencyDiscrete> *WStringFactor::GetFrecuencyDiscrete(void)
{
    return(GetDiscreteFrecuency<std::wstring>(cells));
}

Variant WStringFactor::GetCell(uint index)
{
    Variant variant(cells[index]);

    return(variant);
}

//------------------------------------------------------------------------| DataFrame

DataFrame::DataFrame(void) {}

ubyte DataFrame::GetColumnByAttribute(const std::wstring &attribute)
{
    for(ubyte i = 0, n = factors.size(); i < n; ++i)
    {
        if(factors[i]->attribute == attribute)
            return(i);
    }

    return(-1);
}

ubyte DataFrame::GetColumnType(ubyte index)
{
    BoolFactor *boolFactor = dynamic_cast<BoolFactor *>(factors[index]);

    if(boolFactor) return(BoolType);

    IntFactor *intFactor = dynamic_cast<IntFactor *>(factors[index]);

    if(intFactor) return(IntType);

    FloatFactor *floatFactor = dynamic_cast<FloatFactor *>(factors[index]);

    if(floatFactor) return(FloatType);

    WStringFactor *wstringFactor = dynamic_cast<WStringFactor *>(factors[index]);

    if(wstringFactor) return(WStringType);

    return(GenericType);
}

DataFrame *DataFrame::GetSubDataFrame(const std::vector <uint> &indexes)
{
    DataFrame *dataframe = new DataFrame();

    for(ubyte i = 0, n = factors.size(); i < n; ++i)
    {
        Factor *factor = nullptr;

        switch(GetColumnType(i))
        {
        case BoolType :
        {
            BoolFactor *source = dynamic_cast<BoolFactor *>(factors[i]);
            BoolFactor *target = new BoolFactor(*source);

            for(uint j = 0, m = target->cells.size(); j < m; ++j)
            {
                if(std::find(indexes.begin(), indexes.end(), m - j - 1) == indexes.end())
                    target->cells.erase(target->cells.begin() + m - j - 1);
            }

            factor = target;

            break;
        }
        case IntType :
        {
            IntFactor *source = dynamic_cast<IntFactor *>(factors[i]);
            IntFactor *target = new IntFactor(*source);

            for(uint j = 0, m = target->cells.size(); j < m; ++j)
            {
                if(std::find(indexes.begin(), indexes.end(), m - j - 1) == indexes.end())
                    target->cells.erase(target->cells.begin() + m - j - 1);
            }

            factor = target;

            break;
        }
        case FloatType :
        {
            FloatFactor *source = dynamic_cast<FloatFactor *>(factors[i]);
            FloatFactor *target = new FloatFactor(*source);

            for(uint j = 0, m = target->cells.size(); j < m; ++j)
            {
                if(std::find(indexes.begin(), indexes.end(), m - j - 1) == indexes.end())
                    target->cells.erase(target->cells.begin() + m - j - 1);
            }

            factor = target;

            break;
        }
        case WStringType :
        {
            WStringFactor *source = dynamic_cast<WStringFactor *>(factors[i]);
            WStringFactor *target = new WStringFactor(*source);

            for(uint j = 0, m = target->cells.size(); j < m; ++j)
            {
                if(std::find(indexes.begin(), indexes.end(), m - j - 1) == indexes.end())
                    target->cells.erase(target->cells.begin() + m - j - 1);
            }

            factor = target;

            break;
        }
        }

        dataframe->factors.push_back(factor);
    }

    return(dataframe);
}
