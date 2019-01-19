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

bool Variant::operator<(const Variant &rhs) const
{
    switch(type)
    {
    case Bool : return(data.b < rhs.data.b);
    case Int : return(data.i < rhs.data.i);
    case Float : return(data.f < rhs.data.f);
    case WString : return(GetWString() < rhs.GetWString());
    }

    return(false);
}

bool Variant::operator<=(const Variant &rhs) const
{
    switch(type)
    {
    case Bool : return(data.b <= rhs.data.b);
    case Int : return(data.i <= rhs.data.i);
    case Float : return(data.f <= rhs.data.f);
    case WString : return(GetWString() <= rhs.GetWString());
    }

    return(false);
}

bool Variant::operator>=(const Variant &rhs) const
{
    switch(type)
    {
    case Bool : return(data.b >= rhs.data.b);
    case Int : return(data.i >= rhs.data.i);
    case Float : return(data.f >= rhs.data.f);
    case WString : return(GetWString() >= rhs.GetWString());
    }

    return(false);
}

bool Variant::operator>(const Variant &rhs) const
{
    switch(type)
    {
    case Bool : return(data.b > rhs.data.b);
    case Int : return(data.i > rhs.data.i);
    case Float : return(data.f > rhs.data.f);
    case WString : return(GetWString() > rhs.GetWString());
    }

    return(false);
}

bool Variant::IsNull(void)
{
    switch(type)
    {
    case Bool : return(false);
    case Int : return(data.i == 0);
    case Float : return(data.f == 0.0f);
    case WString : return(GetWString().empty());
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
    case WString : return(GetWString());
    case Generic : return(L"");
    }

    return(L"");
}

std::wstring Variant::GetWString(void) const
{
    std::wstring *wstring = static_cast<std::wstring *>(data.ptr);

    return(*wstring);
}

//------------------------------------------------------------------------| Attribute

Attribute::Attribute(const std::wstring &attribute, const bool discrete) :
    name(attribute), discrete(discrete) {}

uint Attribute::Size(void) {return(0);}

bool Attribute::GetUniformity(void) {return(true);}

Variant Attribute::GetMode(const std::vector<uint> &indexes)
{
    Variant variant(L"");

    return(variant);
}

float Attribute::GetAttributeEntropy(const std::vector<uint> &/*indexes*/) {return(0.0f);}

float Attribute::GetAttributeGiniIndex(const std::vector<uint> &/*indexes*/) {return(1.0f);}

std::vector<Attribute::ProbabilityDistribution> *Attribute::GetProbabilityDistribution(
    const std::vector<uint> &/*restriction*/) {return(nullptr);}

Variant Attribute::GetCell(uint index)
{
    Variant variant(L"");

    return(variant);
}

//------------------------------------------------------------------------| BoolAttribute

BoolAttribute::BoolAttribute(const std::wstring &attribute) : Attribute(attribute, true) {}

uint BoolAttribute::Size(void) {return(cells.size());}

bool BoolAttribute::GetUniformity(void)
{
    for(uint i = 1, n = cells.size(); i < n; ++i)
    {
        if(cells[i] != cells.front())
            return(false);
    }

    return(true);
}

Variant BoolAttribute::GetMode(const std::vector<uint> &indexes)
{
    std::map<bool, int> frecuency = GetFrecuencyMapping<bool>(cells, indexes);

    Variant variant(FrecuencyMode<bool>(frecuency));

    return(variant);
}

float BoolAttribute::GetAttributeEntropy(const std::vector<uint> &restrictions)
{
    return(GetEntropy(cells, restrictions));
}

float BoolAttribute::GetAttributeGiniIndex(const std::vector<uint> &restrictions)
{
    return(GetGiniIndex(cells, restrictions));
}

std::vector<Attribute::ProbabilityDistribution> *BoolAttribute::GetProbabilityDistribution(
    const std::vector<uint> &restriction)
{
    return(GetDistributionFuncion<bool>(cells, restriction));
}

Variant BoolAttribute::GetCell(uint index)
{
    Variant variant(cells[index]);

    return(variant);
}

//------------------------------------------------------------------------| IntAttribute

IntAttribute::IntAttribute(const std::wstring &attribute) : Attribute(attribute, false) {}

uint IntAttribute::Size(void) {return(cells.size());}

bool IntAttribute::GetUniformity(void)
{
    for(uint i = 1, n = cells.size(); i < n; ++i)
    {
        if(cells[i] != cells.front())
            return(false);
    }

    return(true);
}

Variant IntAttribute::GetMode(const std::vector<uint> &indexes)
{
    std::map<int, int> frecuency = GetFrecuencyMapping<int>(cells, indexes);

    Variant variant(FrecuencyMode<int>(frecuency));

    return(variant);
}

float IntAttribute::GetAttributeEntropy(const std::vector<uint> &restrictions)
{
    return(GetEntropy(cells, restrictions));
}

float IntAttribute::GetAttributeGiniIndex(const std::vector<uint> &restrictions)
{
    return(GetGiniIndex(cells, restrictions));
}

std::vector<Attribute::ProbabilityDistribution> *IntAttribute::GetProbabilityDistribution(
    const std::vector<uint> &restriction)
{
    if(discrete)
        return(GetDistributionFuncion<int>(cells, restriction));
    else
        return(GetDensityFunction<int>(cells, restriction));
}

Variant IntAttribute::GetCell(uint index)
{
    Variant variant(cells[index]);

    return(variant);
}

//------------------------------------------------------------------------| FloaAttribute

FloaAttribute::FloaAttribute(const std::wstring &attribute) : Attribute(attribute, false) {}

uint FloaAttribute::Size(void) {return(cells.size());}

bool FloaAttribute::GetUniformity(void)
{
    for(uint i = 1, n = cells.size(); i < n; ++i)
    {
        if(cells[i] != cells.front())
            return(false);
    }

    return(true);
}

Variant FloaAttribute::GetMode(const std::vector<uint> &indexes)
{
    std::map<float, int> frecuency = GetFrecuencyMapping<float>(cells, indexes);

    Variant variant(FrecuencyMode<float>(frecuency));

    return(variant);
}

float FloaAttribute::GetAttributeEntropy(const std::vector<uint> &restrictions)
{
    return(GetEntropy(cells, restrictions));
}

float FloaAttribute::GetAttributeGiniIndex(const std::vector<uint> &restrictions)
{
    return(GetGiniIndex(cells, restrictions));
}

std::vector<Attribute::ProbabilityDistribution> *FloaAttribute::GetProbabilityDistribution(
    const std::vector<uint> &restriction)
{
    if(discrete)
        return(GetDistributionFuncion<float>(cells, restriction));
    else
        return(GetDensityFunction<float>(cells, restriction));
}

Variant FloaAttribute::GetCell(uint index)
{
    Variant variant(cells[index]);

    return(variant);
}

//------------------------------------------------------------------------| WStringAttribute

WStringAttribute::WStringAttribute(const std::wstring &attribute) : Attribute(attribute, true) {}

uint WStringAttribute::Size(void) {return(cells.size());}

bool WStringAttribute::GetUniformity(void)
{
    for(uint i = 1, n = cells.size(); i < n; ++i)
    {
        if(cells[i] != cells.front())
            return(false);
    }

    return(true);
}

Variant WStringAttribute::GetMode(const std::vector<uint> &indexes)
{
    std::map<std::wstring, int> frecuency = GetFrecuencyMapping<std::wstring>(cells, indexes);

    Variant variant(FrecuencyMode<std::wstring>(frecuency));

    return(variant);
}

float WStringAttribute::GetAttributeEntropy(const std::vector<uint> &restrictions)
{
    return(GetEntropy(cells, restrictions));
}

float WStringAttribute::GetAttributeGiniIndex(const std::vector<uint> &restrictions)
{
    return(GetGiniIndex(cells, restrictions));
}

std::vector<Attribute::ProbabilityDistribution> *WStringAttribute::GetProbabilityDistribution(
    const std::vector<uint> &restriction)
{
    return(GetDistributionFuncion<std::wstring>(cells, restriction));
}

Variant WStringAttribute::GetCell(uint index)
{
    Variant variant(cells[index]);

    return(variant);
}

//------------------------------------------------------------------------| DataFrame

DataFrame::DataFrame(void) {}

uint DataFrame::Size(void)
{
    uint size = std::numeric_limits<int>::max();

    for(uint i = 0, n = attributes.size(); i < n; ++i)
        size = min(size, attributes[i]->Size());

    return(size);
}

void DataFrame::Clear(void)
{
    clrptrvector<Attribute *>(attributes);
}

ubyte DataFrame::GetColumnByAttribute(const std::wstring &attribute)
{
    for(ubyte i = 0, n = attributes.size(); i < n; ++i)
    {
        if(attributes[i]->name == attribute)
            return(i);
    }

    return(-1);
}

ubyte DataFrame::GetColumnType(ubyte index)
{
    BoolAttribute *boolFactor = dynamic_cast<BoolAttribute *>(attributes[index]);

    if(boolFactor) return(BoolType);

    IntAttribute *intFactor = dynamic_cast<IntAttribute *>(attributes[index]);

    if(intFactor) return(IntType);

    FloaAttribute *floatFactor = dynamic_cast<FloaAttribute *>(attributes[index]);

    if(floatFactor) return(FloatType);

    WStringAttribute *wstringFactor = dynamic_cast<WStringAttribute *>(attributes[index]);

    if(wstringFactor) return(WStringType);

    return(GenericType);
}

DataFrame *DataFrame::GetSubDataFrame(const std::vector <uint> &indexes)
{
    DataFrame *dataframe = new DataFrame();

    for(ubyte i = 0, n = attributes.size(); i < n; ++i)
    {
        Attribute *factor = nullptr;

        switch(GetColumnType(i))
        {
        case BoolType :
        {
            BoolAttribute *source = dynamic_cast<BoolAttribute *>(attributes[i]);
            BoolAttribute *target = new BoolAttribute(*source);

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
            IntAttribute *source = dynamic_cast<IntAttribute *>(attributes[i]);
            IntAttribute *target = new IntAttribute(*source);

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
            FloaAttribute *source = dynamic_cast<FloaAttribute *>(attributes[i]);
            FloaAttribute *target = new FloaAttribute(*source);

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
            WStringAttribute *source = dynamic_cast<WStringAttribute *>(attributes[i]);
            WStringAttribute *target = new WStringAttribute(*source);

            for(uint j = 0, m = target->cells.size(); j < m; ++j)
            {
                if(std::find(indexes.begin(), indexes.end(), m - j - 1) == indexes.end())
                    target->cells.erase(target->cells.begin() + m - j - 1);
            }

            factor = target;

            break;
        }
        }

        dataframe->attributes.push_back(factor);
    }

    return(dataframe);
}

//------------------------------------------------------------------------| Common

bool ML::Validate(Variant &a, MathOp &mathop, Variant &b)
{
    switch(mathop)
    {
    case 0 : return(a == b);
    case 1 : return(a < b);
    case 2 : return(a <= b);
    case 3 : return(a >= b);
    case 4 : return(a > b);
    }

    return(false);
}
