#![recursion_limit = "128"]

/*!
Derive `trait Serializable` for a structure.

Using this crate, it is enough to `#[derive(Serializable)]` on a struct to be able to serialize and
deserialize it. All the fields must implement [`Serializable`](../npy/trait.Serializable.html).

*/

extern crate proc_macro;
extern crate syn;
#[macro_use]
extern crate quote;

use proc_macro::TokenStream;
use quote::{ToTokens, Tokens};
use syn::Data;

/// Macros 1.1-based custom derive function
#[proc_macro_derive(Serializable)]
pub fn npy_data(input: TokenStream) -> TokenStream {
    // Construct a string representation of the type definition
    // let s = input.to_string();

    // Parse the string representation
    let ast = syn::parse(input).unwrap();

    // Build the impl
    let expanded = impl_npy_data(&ast);

    // Return the generated impl
    expanded.into()
}

fn impl_npy_data(ast: &syn::DeriveInput) -> quote::Tokens {
    let name = &ast.ident;
    let fields = match ast.data {
        Data::Struct(ref data) => &data.fields,
        _ => panic!("#[derive(Serializable)] can only be used with structs"),
    };
    // Helper is provided for handling complex generic types correctly and effortlessly
    let (impl_generics, ty_generics, where_clause) = ast.generics.split_for_impl();

    let idents = fields
        .iter()
        .map(|f| {
            let mut t = Tokens::new();
            f.ident
                .clone()
                .expect("Tuple structs not supported")
                .to_tokens(&mut t);
            t
        })
        .collect::<Vec<_>>();
    let types = fields
        .iter()
        .map(|f| {
            let mut t = Tokens::new();
            f.ty.to_tokens(&mut t);
            t
        })
        .collect::<Vec<_>>();

    let idents_c = idents.clone();
    let idents_str = idents
        .clone()
        .into_iter()
        .map(|t| t.to_string())
        .collect::<Vec<_>>();
    let idents_str_c1 = idents_str.clone();
    let types_c1 = types.clone();
    let types_c2 = types.clone();
    let types_c3 = types.clone();

    let nats_0 = 0usize..;
    let nats_1 = 0usize..;
    let n_fields = types.len();

    quote! {
        impl #impl_generics ::npy::Serializable for #name #ty_generics #where_clause {
            fn dtype() -> ::npy::DType {
                ::npy::DType::Record(vec![#(
                    ::npy::Field {
                        name: #idents_str_c1.to_string(),
                        dtype: <#types_c1 as ::npy::Serializable>::dtype()
                    }
                ),*])
            }

            fn n_bytes() -> usize {
                #( <#types_c2 as ::npy::Serializable>::n_bytes() )+*
            }

            #[allow(unused_assignments)]
            fn read(buf: &[u8]) -> Self {
                let mut offset = 0;
                let mut offsets = [0; #n_fields + 1];
                #(
                    offset += <#types_c3 as ::npy::Serializable>::n_bytes();
                    offsets[#nats_0 + 1] = offset;
                )*

                #name { #(
                    #idents: ::npy::Serializable::read(&buf[offsets[#nats_1]..])
                ),* }
            }

            fn write<W: ::std::io::Write>(&self, writer: &mut W) -> ::std::io::Result<()> {
                #( ::npy::Serializable::write(&self.#idents_c, writer)?; )*
                Ok(())
            }
        }
    }
}
