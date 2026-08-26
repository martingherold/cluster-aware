#pragma once

#include <scip/scip.h>

#include <exception>
#include <new>
#include <utility>

namespace cluster_aware::detail {

template <typename Callback>
SCIP_RETCODE invoke_scip_callback(const char* operation, Callback&& callback) noexcept
{
    try {
        return std::forward<Callback>(callback)();
    } catch (const std::bad_alloc&) {
        SCIPerrorMessage("%s failed: out of memory\n", operation);
        return SCIP_NOMEMORY;
    } catch (const std::exception& exception) {
        SCIPerrorMessage("%s failed: %s\n", operation, exception.what());
        return SCIP_ERROR;
    } catch (...) {
        SCIPerrorMessage("%s failed with an unknown C++ exception\n", operation);
        return SCIP_ERROR;
    }
}

} // namespace cluster_aware::detail
