def create(v_n, u_n, c_n, x_inf, x_sup):
    BNF_grammar = {
        "<expr>": [["<C>", "[", "<P>", "]"], ["<C>", "[", "<P>", "]", "<expr>"],
                   ['<C>', '[', '<P>', ']', '<expr>', '<expr>']],
        "<P>": [["<V>", "<U>", "<X>"]],
        "<C>": [],
        "<V>": [],
        "<U>": [],
        "<X>": []
    }

    BNF_grammar["<C>"] = [[f"l{i + 1}"] for i in range(c_n)]
    BNF_grammar["<V>"] = [[f"v{i + 1}"] for i in range(v_n)]
    BNF_grammar["<U>"] = [[f"u{i + 1}"] for i in range(u_n)]
    BNF_grammar["<X>"] = [[f"{i + 1}"] for i in range(x_inf, x_sup)]

    return BNF_grammar

#print(create(5, 2, 2, 0, 50))