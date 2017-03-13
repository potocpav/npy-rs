#![recursion_limit = "128"]
// #![feature(proc_macro)]

/*!
Derive `trait NpyData` for a structure.

Using this crate, it is enough to `#[derive(NpyData)]` on a struct to be able to serialize and
deserialize it. All the fields must implement [`Serializable`](../npy/trait.Serializable.html).

*/

extern crate proc_macro;
extern crate syn;
#[macro_use]
extern crate quote;

use proc_macro::TokenStream;
use syn::Body;
use quote::{Tokens, ToTokens};

/// Macros 1.1-based custom derive function
#[proc_macro_derive(NpyData)]
pub fn npy_data(input: TokenStream) -> TokenStream {
    // Construct a string representation of the type definition
    let s = input.to_string();

    // Parse the string representation
    let ast = syn::parse_macro_input(&s).unwrap();

    // Build the impl
    let expanded = impl_npy_data(&ast);

    // Return the generated impl
    expanded.parse().unwrap()
}

fn impl_npy_data(ast: &syn::DeriveInput) -> quote::Tokens {
    let name = &ast.ident;
    let fields = match ast.body {
        Body::Enum(_) => panic!("#[derive(NpyData)] can only be used with structs"),
        Body::Struct(ref data) => data.fields(),
    };
    // Helper is provided for handling complex generic types correctly and effortlessly
    let (impl_generics, ty_generics, where_clause) = ast.generics.split_for_impl();

    let idents = fields.iter().map(|f| {
        let mut t = Tokens::new();
        f.ident.clone().expect("Tuple structs not supported").to_tokens(&mut t);
        t
    }).collect::<Vec<_>>();
    let types = fields.iter().map(|f|  {
        let mut t = Tokens::new();
        f.ty.to_tokens(&mut t);
        t
    }).collect::<Vec<_>>();

    let idents_c = idents.clone();
    let idents_str = idents.clone().into_iter().map(|t| t.to_string()).collect::<Vec<_>>();
    let idents_str_c1 = idents_str.clone();
    let types_c1 = types.clone();

    quote! {
        impl #impl_generics ::npy::NpyData for #name #ty_generics #where_clause {
            fn get_dtype() -> Vec<(&'static str, ::npy::DType)> {
                vec![#( {
                    (#idents_str_c1, <#types_c1 as ::npy::Serializable>::dtype())
                } ),*]
            }

            fn read_row(c: &mut ::std::io::Cursor<&[u8]>) -> ::std::io::Result<Self> {
                Ok(#name { #(
                    #idents: { ::npy::Serializable::read(c)? }
                ),* })
            }

            fn write_row<W: ::std::io::Write>(&self, writer: &mut W) -> ::std::io::Result<()> {
                #( ::npy::Serializable::write(&self.#idents_c, writer)?; )*
                Ok(())
            }
        }
    }
}
