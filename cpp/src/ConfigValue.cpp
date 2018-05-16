#include "ConfigValue.h"

ConfigValue::ConfigValue(const std::string& s, ParamType type) {
    type_ = type;
    if(type_ == kUint32_t) {
        val_int_ = std::stoi(s);
    } else if(type_ == kFloatType) {
        val_float_ = std::stod(s);
    } else if(type_ == kString) {
        val_string_ = s;
    } else if(type_ == kBool) {
        val_bool_ = StringToBool(s);
    } else if(type_ == kObjectiveType) {
        val_objective_ = StringToObj(s);
    }
}

ConfigValue::operator uint32_t() const{
    return val_int_;
}

ConfigValue::operator float_type() const{
    return val_float_;
}

ConfigValue::operator std::string() const{
    return val_string_;
}

ConfigValue::operator ObjectiveType() const{
    return val_objective_;
}

ConfigValue::operator bool() const{
    return val_bool_;
}

ParamType ConfigValue::GetType() {
    return type_;
}

ObjectiveType ConfigValue::StringToObj(const std::string& s) {
    if(s == "mse") {
        return kMse;
    } else if(s == "logloss") {
        return kLogLoss;
    } else {
        throw std::invalid_argument("Wrong argument for objective");
    }
};

bool ConfigValue::StringToBool(const std::string& s) {
    if(s == "1") {
        return true;
    } else if(s == "0") {
        return false;
    } else {
        throw std::invalid_argument("Ivalid argument for file_has_target");
    }
};