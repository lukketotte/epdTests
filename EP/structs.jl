module Structs

export ConvergenceError

struct ConvergenceError <:Exception
    msg::String
end

end
