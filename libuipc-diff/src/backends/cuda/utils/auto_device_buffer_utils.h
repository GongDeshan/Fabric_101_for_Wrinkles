#pragma once
#include <type_define.h>
#include <uipc/common/span.h>
#include <uipc/backend/buffer_view.h>
#include <muda/buffer.h>
#include <uipc/builtin/backend_name.h>

namespace uipc::backend::cuda
{
namespace detail
{
    template <typename T>
    class AutoBufferView
    {
      public:
        AutoBufferView(muda::DeviceBuffer<T>& buffer)
            : m_buffer(buffer)
        {
        }

      private:
        muda::DeviceBuffer<T>& m_buffer;

      public:
        span<T> as_span(const backend::BufferView& view)
        {
            UIPC_ASSERT(view.element_size() == sizeof(T), "element size mismatch");
            return span<T>{reinterpret_cast<T*>(view.handle()), view.size()};
        }

        muda::BufferView<T> as_buffer_view(const backend::BufferView& view)
        {
            UIPC_ASSERT(view.element_size() == sizeof(T), "element size mismatch");
            return muda::BufferView<T>{reinterpret_cast<T*>(view.handle()), view.size()};
        }

        muda::BufferView<T> initialized_buffer(const backend::BufferView& input_view)
        {
            if(input_view.backend() == builtin::none)
            {
                m_buffer.resize(input_view.size());
                auto s           = as_span(input_view);
                auto buffer_view = m_buffer.view();
                buffer_view.copy_from(s.data());
                return buffer_view;
            }
            else if(input_view.backend() == builtin::cuda)
            {
                return as_buffer_view(input_view);
            }
            else
            {
                UIPC_ASSERT(false, "unsupported backend");
            }

            return muda::BufferView<T>{};  // unreachable
        }

        muda::BufferView<T> uninitialized_buffer(const backend::BufferView& input_view)
        {
            if(input_view.backend() == builtin::none)
            {
                m_buffer.resize(input_view.size());
                return m_buffer.view();
            }
            else if(input_view.backend() == builtin::cuda)
            {
                return as_buffer_view(input_view);
            }
            else
            {
                UIPC_ASSERT(false, "unsupported backend");
            }

            return muda::BufferView<T>{};  // unreachable
        }

        void sync_data(backend::BufferView view)
        {
            if(view.backend() == builtin::none)
            {
                auto s = as_span(view);
                m_buffer.view().copy_to(s.data());
            }
        }
    };

}  // namespace detail


template <typename T>
class InArgBufferView
{
  public:
    InArgBufferView(muda::DeviceBuffer<T>& buffer)
        : m_impl{buffer}
    {
    }

    muda::CBufferView<T> operator()(backend::BufferView view)
    {
        return m_impl.initialized_buffer(view);
    }

  private:
    detail::AutoBufferView<T> m_impl;
};

// CTAD
template <typename T>
InArgBufferView(muda::DeviceBuffer<T>&) -> InArgBufferView<T>;

template <typename T>
class OutArgBufferView
{
  public:
    OutArgBufferView(muda::DeviceBuffer<T>& buffer)
        : m_impl{buffer}
    {
    }

    muda::BufferView<T> operator()(backend::BufferView view)
    {
        m_view = view;
        return m_impl.uninitialized_buffer(view);
    }

    ~OutArgBufferView() { m_impl.sync_data(m_view); }

  private:
    detail::AutoBufferView<T> m_impl;
    backend::BufferView       m_view;
};

// CTAD
template <typename T>
OutArgBufferView(muda::DeviceBuffer<T>&) -> OutArgBufferView<T>;

template <typename T>
class InOutArgBufferView
{
  public:
    InOutArgBufferView(muda::DeviceBuffer<T>& buffer)
        : m_impl{buffer}
    {
    }

    muda::BufferView<T> operator()(backend::BufferView view)
    {
        m_view = view;
        return m_impl.initialized_buffer(view);
    }

    ~InOutArgBufferView() { m_impl.sync_data(m_view); }

  private:
    detail::AutoBufferView<T> m_impl;
    backend::BufferView       m_view;
};

// CTAD
template <typename T>
InOutArgBufferView(muda::DeviceBuffer<T>&) -> InOutArgBufferView<T>;
}  // namespace uipc::backend::cuda
