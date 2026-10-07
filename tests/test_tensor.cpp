#include "test_framework.h"

#include <forge/tensor.h>

#include <sstream>

using namespace forge;

TEST(tensor_contiguous_strides) {
    CHECK(contiguous_strides({2, 3, 4}) == (Shape{12, 4, 1}));
    CHECK(contiguous_strides({5}) == (Shape{1}));
    CHECK(contiguous_strides({}).empty());
}

TEST(tensor_numel_and_scalar) {
    CHECK(numel({2, 3, 4}) == 24);
    auto s = Tensor::full({}, 5.0f);
    CHECK(s.ndim() == 0);
    CHECK(s.numel() == 1);
    CHECK(s.item() == 5.0f);
}

TEST(tensor_factories) {
    auto z = Tensor::zeros({2, 2});
    auto o = Tensor::ones({3});
    CHECK(z.at({1, 1}) == 0.0f);
    CHECK(o.at({2}) == 1.0f);
    auto r = Tensor::arange(0, 2, 0.5f);
    CHECK(r.numel() == 4);
    CHECK(r.at({3}) == 1.5f);
    auto e = Tensor::eye(3);
    CHECK(e.at({1, 1}) == 1.0f && e.at({0, 1}) == 0.0f);
    CHECK_THROWS(Tensor::from_vector({1, 2, 3}, {2, 2}));
    CHECK_THROWS(Tensor::arange(0, 1, 0));
}

TEST(tensor_random_is_reproducible) {
    manual_seed(7);
    auto a = Tensor::randn({4});
    manual_seed(7);
    auto b = Tensor::randn({4});
    for (std::size_t i = 0; i < 4; ++i) CHECK(a.at({i}) == b.at({i}));
    auto u = Tensor::rand({1000});
    for (std::size_t i = 0; i < 1000; ++i) CHECK(u.at({i}) >= 0.0f && u.at({i}) < 1.0f);
}

TEST(tensor_at_and_bounds) {
    auto t = Tensor::from_vector({1, 2, 3, 4, 5, 6}, {2, 3});
    CHECK(t.at({1, 0}) == 4.0f);
    CHECK_THROWS(t.at({2, 0}));
    CHECK_THROWS(t.at({0}));
}

TEST(tensor_copy_shares_data) {
    auto a = Tensor::zeros({2, 2});
    Tensor b = a;
    b.at({0, 0}) = 7.0f;
    CHECK(a.at({0, 0}) == 7.0f);
}

TEST(tensor_views_share_data) {
    auto a = Tensor::arange(0, 6);
    auto m = a.view({2, 3});
    m.at({0, 0}) = 100.0f;
    CHECK(a.at({0}) == 100.0f);
    CHECK_THROWS(a.view({4}));
}

TEST(tensor_transpose_and_contiguous) {
    auto m = Tensor::arange(0, 6).view({2, 3});
    auto t = m.transpose(0, 1);
    CHECK(t.shape() == (Shape{3, 2}));
    CHECK(t.strides() == (Shape{1, 3}));
    CHECK(!t.is_contiguous());
    CHECK(t.at({2, 1}) == 5.0f);
    auto c = t.contiguous();
    CHECK(c.is_contiguous());
    CHECK(c.data()[1] == 3.0f);
    CHECK_THROWS(t.view({6}));
    auto r = t.reshape({6});
    CHECK(r.at({1}) == 3.0f);
}

TEST(tensor_permute) {
    auto x = Tensor::arange(0, 24).view({2, 3, 4});
    auto p = x.permute({2, 0, 1});
    CHECK(p.shape() == (Shape{4, 2, 3}));
    CHECK(p.at({3, 1, 2}) == x.at({1, 2, 3}));
    CHECK_THROWS(x.permute({0, 0, 1}));
}

TEST(tensor_expand_uses_zero_stride) {
    auto b = Tensor::from_vector({10, 20, 30}, {3});
    auto e = b.expand({2, 3});
    CHECK(e.strides() == (Shape{0, 1}));
    CHECK(e.at({1, 2}) == 30.0f);
    CHECK_THROWS(b.expand({2, 4}));
}

TEST(tensor_unsqueeze_squeeze) {
    auto v = Tensor::ones({3});
    CHECK(v.unsqueeze(0).shape() == (Shape{1, 3}));
    CHECK(v.unsqueeze(1).shape() == (Shape{3, 1}));
    CHECK(v.unsqueeze(1).squeeze(1).shape() == (Shape{3}));
    CHECK_THROWS(v.squeeze(0));
}

TEST(tensor_clone_is_independent) {
    auto a = Tensor::ones({2});
    auto c = a.clone();
    c.at({0}) = 5.0f;
    CHECK(a.at({0}) == 1.0f);
}

TEST(tensor_printing) {
    std::ostringstream os;
    os << Tensor::from_vector({1, 2, 3, 4}, {2, 2});
    CHECK(os.str() == "tensor([[1, 2],\n        [3, 4]])");
    std::ostringstream os2;
    os2 << Tensor();
    CHECK(os2.str() == "tensor(undefined)");
}

TEST(tensor_undefined_throws) {
    Tensor t;
    CHECK(!t.defined());
    CHECK_THROWS(t.shape());
}
