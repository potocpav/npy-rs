#![recursion_limit = "128"]
#![feature(proc_macro)]
extern crate proc_macro;
extern crate syn;
#[macro_use]
extern crate quote;

use proc_macro::TokenStream;
use syn::Body;
use quote::{Tokens, ToTokens};

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
        t.to_string()
    });

    let idents_c = idents.clone();
    let idents_str = idents.clone().into_iter().map(|t| t.to_string());

    quote! {
        impl #impl_generics ::npy::NpyData for #name #ty_generics #where_clause {
            fn get_fields() -> Vec<(&'static str, &'static str)> {
                vec![#( (#idents_str, #types) ),*]
            }

            fn read_row(c: &mut npy::Cursor<&[u8]>) -> Option<Self> {
                Some(Self { #(
                    #idents: {
                        if let Ok(v) = npy::Readable::read(c) {
                            v
                        } else {
                            return None;
                        }
                    }
                ),* })
            }

            fn write_row<W: ::std::io::Write>(&self, writer: &mut W) -> ::std::io::Result<()> {
                #( npy::Writeable::write(self.#idents_c, writer)?; )*
                Ok(())
            }
        }
    }
}
